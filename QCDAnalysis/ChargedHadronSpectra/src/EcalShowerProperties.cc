#include "QCDAnalysis/ChargedHadronSpectra/interface/EcalShowerProperties.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

using namespace std;

const double R_M = 2.2;
//const double R_M = 4.;

/*****************************************************************************/
EcalShowerProperties::EcalShowerProperties
  (const edm::Event & ev, const edm::EventSetup & es)
{
  // Get magnetic field
  edm::ESHandle<MagneticField> theMagneticFieldHandle;
  es.get<IdealMagneticFieldRecord>().get(theMagneticFieldHandle);
  theMagneticField = theMagneticFieldHandle.product();

  // Get propagator
  edm::ESHandle<Propagator> thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",
                                          thePropagatorHandle);
  thePropagator = thePropagatorHandle.product();

  // Get calorimetry
  edm::ESHandle<CaloGeometry> theCaloGeometryHandle;
  es.get<IdealGeometryRecord>().get(theCaloGeometryHandle);
  theCaloGeometry = (const CaloGeometry*)theCaloGeometryHandle.product();

  // Get ecal rechits
  ev.getByLabel("ecalRecHit", "EcalRecHitsEB", recHitsBarrel);
  ev.getByLabel("ecalRecHit", "EcalRecHitsEE", recHitsEndcap);
}

/*****************************************************************************/
FreeTrajectoryState EcalShowerProperties::getTrajectoryAtOuterPoint
  (const reco::Track & track)
{
  GlobalPoint  pos(track.outerX() , track.outerY() , track.outerZ() );
  GlobalVector mom(track.outerPx(), track.outerPy(), track.outerPz());

  GlobalTrajectoryParameters gtp(pos,mom, track.charge(), theMagneticField);

  return FreeTrajectoryState(gtp, track.outerStateCovariance());
}

/*****************************************************************************/
Plane::PlanePointer EcalShowerProperties::getSurface
  (const CaloCellGeometry* cell, int i)
{
  int j = i * 4;

  // Get corners
  const CaloCellGeometry::CornersVec & c(cell->getCorners());

  // Get center
  GlobalPoint center( (c[j].x() + c[j+1].x() + c[j+2].x() + c[j+3].x()) / 4,
                      (c[j].y() + c[j+1].y() + c[j+2].y() + c[j+3].y()) / 4,
                      (c[j].z() + c[j+1].z() + c[j+2].z() + c[j+3].z()) / 4);

  // Get plane
  Surface::PositionType pos(center);
  Surface::RotationType rot(c[j+1]-c[j], c[j+3]-c[j]);
  return Plane::build(pos, rot);
}

/*****************************************************************************/
vector<TrajectoryStateOnSurface> EcalShowerProperties::getEndpoints
  (const FreeTrajectoryState & ftsAtLastPoint,
   const TrajectoryStateOnSurface & tsosBeforeEcal, int subDet)
{
  std::vector<TrajectoryStateOnSurface> tsosEnds;

  const CaloSubdetectorGeometry* geom =
    theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,subDet);

  // Get closest cell 
  DetId detId(geom->getClosestCell(ftsAtLastPoint.position()));

  if(!geom->present(detId)) return tsosEnds;

  const CaloCellGeometry* cell = geom->getGeometry(detId);
  const CaloCellGeometry* oldcell;
  TrajectoryStateOnSurface tsos;

  // Front and back
  for(int i = 0; i < 2; i++)
  {
    int step = 0;

    do
    {
      oldcell = cell;

      Plane::PlanePointer thePlane  = getSurface(oldcell, i);
      tsos = thePropagator->propagate(ftsAtLastPoint,*thePlane);

      if(!tsos.isValid()) return tsosEnds;

      detId = geom->getClosestCell(tsos.globalPosition());

      if(!geom->present(detId)) return tsosEnds;
      cell  = geom->getGeometry(detId);
    }
    while(cell != oldcell && step++ < 5);

    if(step++ < 5) 
      tsosEnds.push_back(tsos);
  }

  return tsosEnds;
}

/*****************************************************************************/
double EcalShowerProperties::getDistance
  (const std::vector<TrajectoryStateOnSurface> & tsosEnds,
   const CaloCellGeometry* cell)
{
  double dmin = 1e+9;
  LocalPoint p;

  // Project to entry surface, 'p1p2' segment
  p = tsosEnds[0].surface().toLocal(tsosEnds[0].globalPosition());
  LocalPoint p1(p.x(), p.y(), 0);

  p = tsosEnds[0].surface().toLocal(tsosEnds[1].globalPosition());
  LocalPoint p2(p.x(), p.y(), 0);

  const CaloCellGeometry::CornersVec & c(cell->getCorners());

  // Project to entry surface, 'c' corners
  for(int i = 0; i < 4; i++)
  {
    p = tsosEnds[0].surface().toLocal(GlobalPoint(c[i].x(),c[i].y(),c[i].z()));
    LocalPoint c(p.x(), p.y(), 0);

    // Calculate distance of 'c' from endpoints 'p1' and 'p2'
    double d1 = (p1 - c).mag2(); // distance from end
    double d2 = (p2 - c).mag2(); // distance from end
    double dm = min(d1,d2);

    // distance from segment
    double lambda = (c - p1) * (p2 - p1) / (p2 - p1).mag2();
    if(lambda > 0 && lambda < 1)
    {
      double dp = (c - p1 - lambda * (p2 - p1)).mag2();
      dm = min (dm,dp);
    }

    dmin = min(dm, dmin);
  }

  return(sqrt(dmin));
}

/*****************************************************************************/
pair<double,double> EcalShowerProperties::processEcalRecHits
  (const std::vector<TrajectoryStateOnSurface> & tsosEnds, int subDet, int & ntime)
{
  const CaloSubdetectorGeometry* geom =
    theCaloGeometry->getSubdetectorGeometry(DetId::Ecal,subDet);

  std::vector<DetId> detIds;
  detIds.push_back(geom->getClosestCell(tsosEnds[0].globalPosition()));
  detIds.push_back(geom->getClosestCell(tsosEnds[1].globalPosition()));

  double energy = 0, time = 0;
  ntime = 0;

  if(subDet == EcalBarrel)
  {
    EBDetId frontId(detIds[0]);
    EBDetId  backId(detIds[1]);

    double ieta =     (frontId.ieta() + backId.ieta())/2.;
    double weta = fabs(frontId.ieta() - backId.ieta())/2.;

    double iphi =     (frontId.iphi() + backId.iphi())/2.;
    double wphi = fabs(frontId.iphi() - backId.iphi())/2.;

    for(EBRecHitCollection::const_iterator recHit = recHitsBarrel->begin();
                                           recHit!= recHitsBarrel->end();
                                           recHit++)
    {
      EBDetId detId(recHit->id());
      const CaloCellGeometry* cell = geom->getGeometry(detId);

      if(fabs(detId.ieta() - ieta) < weta + 4 &&
         fabs(detId.iphi() - iphi) < wphi + 4)
      if(getDistance(tsosEnds, cell) < R_M)
      {
        energy += recHit->energy();
        time   += recHit->energy() * recHit->time();
        ntime++;
      }
    }
  }
  else
  {
    EEDetId frontId(detIds[0]);
    EEDetId  backId(detIds[1]);

    double ix =     (frontId.ix() + backId.ix())/2.;
    double wx = fabs(frontId.ix() - backId.ix())/2.;

    double iy =     (frontId.iy() + backId.iy())/2.;
    double wy = fabs(frontId.iy() - backId.iy())/2.;

    for(EERecHitCollection::const_iterator recHit = recHitsEndcap->begin();
                                           recHit!= recHitsEndcap->end();
                                           recHit++)
    {
      EEDetId detId(recHit->id());
      const CaloCellGeometry* cell = geom->getGeometry(detId);

      if(detId.zside() == frontId.zside() &&
         detId.zside() ==  backId.zside())
      if( fabs(detId.ix() - ix) < wx + 4 &&
          fabs(detId.iy() - iy) < wy + 4)
      if(getDistance(tsosEnds, cell) < R_M)
      {
        energy += recHit->energy();
        time   += recHit->energy() * recHit->time();
        ntime++;
      }
    }
  }

  if(energy > 0) time /= energy;

  return std::pair<double,double> (energy,time);
}

/*****************************************************************************/
pair<double,double> EcalShowerProperties::processTrack
  (const reco::Track & track, int & ntime)
{
  // Get trajectory at outer point
  FreeTrajectoryState ftsAtLastPoint = getTrajectoryAtOuterPoint(track);

  // Ecal cylinder
  double radius  = 129.0; // cm
  double z       = 320.9; // cm
  Surface::RotationType rot;

  // Subdetector
  std::vector<int> subDets;
  subDets.push_back(EcalBarrel);
  subDets.push_back(EcalEndcap);

  std::pair<double,double> result(0,0);

  // Take barrel and endcap
  for(std::vector<int>::const_iterator subDet = subDets.begin();
                                  subDet!= subDets.end(); subDet++)
  {
    TrajectoryStateOnSurface tsosBeforeEcal;

    if(*subDet == EcalBarrel)
    { 
      Surface::PositionType pos(0,0,0);
      Cylinder::CylinderPointer theBarrel = Cylinder::build(radius, pos, rot);
      tsosBeforeEcal = thePropagator->propagate(ftsAtLastPoint, *theBarrel);

      if(!tsosBeforeEcal.isValid())                     continue;
      if(fabs(tsosBeforeEcal.globalPosition().z()) > z) continue;
    }
    else
    {
      Surface::PositionType pos(0,0,z);
      Plane::PlanePointer theEndcap = Plane::build(pos, rot);
      tsosBeforeEcal = thePropagator->propagate(ftsAtLastPoint, *theEndcap);

      if(!tsosBeforeEcal.isValid())                             continue;
      if(fabs(tsosBeforeEcal.globalPosition().perp()) > radius) continue;
    }

    std::vector<TrajectoryStateOnSurface> tsosEnds =
      getEndpoints(ftsAtLastPoint, tsosBeforeEcal, *subDet);

    if(tsosEnds.size() == 2)
      result = processEcalRecHits(tsosEnds, *subDet, ntime);
  }

  return result;
}
