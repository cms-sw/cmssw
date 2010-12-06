/**
 *  \package: MuonIdentification
 *  \class: MuonShowerInformationFiller
 *
 *  Description: class for muon shower identification
 *
 *  $Date: 2010/12/01 09:43:25 $
 *  $Revision: 1.1 $
 *
 *  \author: A. Svyatkovskiy, Purdue University
 *
 **/

#include "RecoMuon/MuonIdentification/interface/MuonShowerInformationFiller.h"

// system include files
#include <memory>
#include <algorithm>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBreaker.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/MuonReco/interface/MuonShower.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace std;
using namespace edm;


//
// Constructor
//
MuonShowerInformationFiller::MuonShowerInformationFiller(const edm::ParameterSet& par) :
  theService(0),
  theDTRecHitLabel(par.getParameter<InputTag>("DTRecSegmentLabel")),
  theCSCSegmentsLabel(par.getParameter<InputTag>("CSCSegmentLabel")),
  theCSCRecHitLabel(par.getParameter<InputTag>("CSCRecSegmentLabel")),
  theDT4DRecSegmentLabel(par.getParameter<InputTag>("DT4DRecSegmentLabel"))
{

  edm::ParameterSet serviceParameters = par.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);

  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");

  theCacheId_TRH = 0;
  theCacheId_MT = 0;

  category_ = "MuonShowerInformationFiller";

}

//
// Destructor
//
MuonShowerInformationFiller::~MuonShowerInformationFiller() {
  if (theService) delete theService;
}

reco::MuonShower MuonShowerInformationFiller::fillShowerInformation( const reco::Muon& muon, const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  reco::MuonShower returnShower;
  
  // Update the services
  theService->update(iSetup);
  setEvent(iEvent);
  setServices(theService->eventSetup());
  
  std::vector<int> nHitsUncorrelated = stationUncorrelatedHits(muon);
  std::vector<double> showerSizeT = stationShowerSizes(muon);
  std::vector<double> showerDeltaR = stationShowerRSizes(muon);
  
  returnShower.nHitsUncorrelated = nHitsUncorrelated; 
  returnShower.showerSizeT = showerSizeT;
  returnShower.showerDeltaR = showerDeltaR; 

  return returnShower;

}

//
// Set Event
//
void MuonShowerInformationFiller::setEvent(const edm::Event& event) {

  // get all the necesary products
  event.getByLabel(theDTRecHitLabel, theDTRecHits);
  event.getByLabel(theCSCSegmentsLabel, theCSCSegments);
  event.getByLabel(theDT4DRecSegmentLabel, theDT4DRecSegments);
  event.getByLabel(theCSCRecHitLabel, theCSCRecHits);

}


//
// Set services
//
void MuonShowerInformationFiller::setServices(const EventSetup& setup) {

  // DetLayer Geometry
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  setup.get<IdealMagneticFieldRecord>().get(theField);
  setup.get<TrackerRecoGeometryRecord>().get(theTracker);
  setup.get<MuonRecoGeometryRecord>().get(theMuonGeometry);

  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  }

}


//
// Get 4D rechits
//
MuonTransientTrackingRecHit::MuonRecHitContainer 
MuonShowerInformationFiller::recHits4D(const GeomDet* geomDet, 
                            edm::Handle<DTRecSegment4DCollection> dtSegments, 
                            edm::Handle<CSCSegmentCollection> cscSegments) const {

  MuonTransientTrackingRecHit::MuonRecHitContainer result;

  DetId geoId = geomDet->geographicalId();

  if (geoId.subdetId() == MuonSubdetId::DT) {

    DTChamberId chamberId(geoId.rawId());
    DTRecSegment4DCollection::range range = dtSegments->get(chamberId);
    for (DTRecSegment4DCollection::const_iterator rechit = range.first;
      rechit!=range.second;++rechit) {
      result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  } 
  else if (geoId.subdetId() == MuonSubdetId::CSC) {

    CSCDetId did(geoId.rawId());
    CSCSegmentCollection::range range = cscSegments->get(did);
    
    for (CSCSegmentCollection::const_iterator rechit = range.first;
      rechit!=range.second;++rechit) {
      result.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet,&*rechit));
    }
  }
  else if (geoId.subdetId() == MuonSubdetId::RPC) {
    LogTrace(category_) << "Wrong subdet id" << endl;
  }

  return result;

}


//
// Number of correlated hits
//
int MuonShowerInformationFiller::numberOfCorrelatedHits(const MuonTransientTrackingRecHit::MuonRecHitContainer& hits4d) const {

  if (hits4d.empty()) return 0;

  TransientTrackingRecHit::ConstRecHitPointer muonRecHit(hits4d.front().get());
  TransientTrackingRecHit::ConstRecHitContainer allhits1dcorrelated = MuonTransientTrackingRecHitBreaker::breakInSubRecHits(muonRecHit,2);

  if (hits4d.size() == 1) return allhits1dcorrelated.size();

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ihit4d = hits4d.begin() + 1;
       ihit4d != hits4d.end(); ++ihit4d) {

    TransientTrackingRecHit::ConstRecHitPointer muonRecHit((*ihit4d).get());
    TransientTrackingRecHit::ConstRecHitContainer hits1 =
    MuonTransientTrackingRecHitBreaker::breakInSubRecHits(muonRecHit,2);

    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit1 = hits1.begin();
         ihit1 != hits1.end(); ++ihit1 ) {

      bool usedbefore = false;       
      DetId thisID = (*ihit1)->geographicalId();
      LocalPoint lp1din4d = (*ihit1)->localPosition();
      GlobalPoint gp1din4d = (*ihit1)->globalPosition();

      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit2 = allhits1dcorrelated.begin();
           ihit2 != allhits1dcorrelated.end(); ++ihit2 ) {

        DetId thisID2 = (*ihit2)->geographicalId();
        LocalPoint lp1din4d2 = (*ihit2)->localPosition();
        GlobalPoint gp1din4d2 = (*ihit2)->globalPosition();

        if ( (gp1din4d2 - gp1din4d).mag() < 1.0 ) usedbefore = true;

      }
      if ( !usedbefore ) allhits1dcorrelated.push_back(*ihit1);
    }
  }

  return allhits1dcorrelated.size();

}


//
// Find cluster
//
MuonTransientTrackingRecHit::MuonRecHitContainer 
MuonShowerInformationFiller::findPhiCluster(MuonTransientTrackingRecHit::MuonRecHitContainer& muonRecHits, 
                              const GlobalPoint& refpoint) const {

  if ( muonRecHits.empty() ) return muonRecHits;

  //clustering step by phi
  float step = 0.05;
  MuonTransientTrackingRecHit::MuonRecHitContainer result;

  stable_sort(muonRecHits.begin(), muonRecHits.end(), AbsLessDPhi(refpoint));

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ihit = muonRecHits.begin(); ihit != muonRecHits.end() - 1; ++ihit) {
      if (fabs(deltaPhi((*(ihit+1))->globalPosition().phi(), (*ihit)->globalPosition().phi() )) < step) {
          result.push_back(*ihit);
        } else {
           break;
       }
  } 

  LogTrace(category_) <<  "phi front: " << muonRecHits.front()->globalPosition().phi() << endl;    
  LogTrace(category_) <<  "phi back: " << muonRecHits.back()->globalPosition().phi() << endl;

  return result;

}

//
//
//
MuonTransientTrackingRecHit::MuonRecHitContainer
MuonShowerInformationFiller::findThetaCluster(MuonTransientTrackingRecHit::MuonRecHitContainer& muonRecHits,
                              const GlobalPoint& refpoint) const {

  if ( muonRecHits.empty() ) return muonRecHits;

  //clustering step by theta
  float step = 0.05;
  MuonTransientTrackingRecHit::MuonRecHitContainer result;

  stable_sort(muonRecHits.begin(), muonRecHits.end(), AbsLessDTheta(refpoint));

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ihit = muonRecHits.begin(); ihit != muonRecHits.end() - 1; ++ihit) {
      if (fabs((*(ihit+1))->globalPosition().theta() - (*ihit)->globalPosition().theta() ) < step) {
          result.push_back(*ihit);
        } else {
           break;
       }
  }

  return result;

}


MuonTransientTrackingRecHit::MuonRecHitContainer
MuonShowerInformationFiller::findPerpCluster(MuonTransientTrackingRecHit::MuonRecHitContainer& muonRecHits) const {

  if ( muonRecHits.empty() ) return muonRecHits;

  stable_sort(muonRecHits.begin(), muonRecHits.end(), LessPerp());

  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator
  seedhit = min_element(muonRecHits.begin(), muonRecHits.end(), LessPerp());

  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ihigh = seedhit;
  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ilow = seedhit;

  float step = 0.1;
  while (ihigh != muonRecHits.end()-1 && ( fabs((*(ihigh+1))->globalPosition().perp() - (*ihigh)->globalPosition().perp() ) < step)  ) {
    ihigh++;
  }
  while (ilow != muonRecHits.begin() && ( fabs((*ilow)->globalPosition().perp() - (*(ilow -1))->globalPosition().perp()) < step ) ) {
    ilow--;
  }

  MuonTransientTrackingRecHit::MuonRecHitContainer result(ilow, ihigh);

  return result;

}

//
// Get compatible dets
//
vector<const GeomDet*> MuonShowerInformationFiller::getCompatibleDets(const reco::Track& track) const {

  vector<const GeomDet*> total;
  total.reserve(1000);
/*
  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.globalTrack();
  else if ( muon.isStandAloneMuon() )    track = muon.outerTrack();
  else return total;
*/
  LogTrace(category_)  << "Consider a track " << track.p() << " eta: " << track.eta() << " phi " << track.phi() << endl;

  TrajectoryStateTransform tsTrans;
  TrajectoryStateOnSurface innerTsos = tsTrans.innerStateOnSurface(track, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface outerTsos = tsTrans.outerStateOnSurface(track, *theService->trackingGeometry(), &*theService->magneticField());

  GlobalPoint innerPos = innerTsos.globalPosition();
  GlobalPoint outerPos = outerTsos.globalPosition();

  vector<GlobalPoint> allCrossingPoints;

  // first take DT
  const vector<DetLayer*>& dtlayers = theService->detLayerGeometry()->allDTLayers();

  for (vector<DetLayer*>::const_iterator iLayer = dtlayers.begin(); iLayer != dtlayers.end(); ++iLayer) {

    // crossing points of track with cylinder
    GlobalPoint xPoint = crossingPoint(innerPos, outerPos, dynamic_cast<const BarrelDetLayer*>(*iLayer));
      
    // check if point is inside the detector
    if ((fabs(xPoint.y()) < 1000.0) && (fabs(xPoint.z()) < 1500 ) && 
             (!(xPoint.y() == 0 && xPoint.x() == 0 && xPoint.z() == 0))) allCrossingPoints.push_back(xPoint);
  }

  stable_sort(allCrossingPoints.begin(), allCrossingPoints.end(), LessMag(innerPos) );

  vector<const GeomDet*> tempDT;

  for (vector<GlobalPoint>::const_iterator ipos = allCrossingPoints.begin(); ipos != allCrossingPoints.end(); ++ipos) {

    tempDT = dtPositionToDets(*ipos);
    vector<const GeomDet*>::const_iterator begin = tempDT.begin();
    vector<const GeomDet*>::const_iterator end = tempDT.end();
     
    for (; begin!=end;++begin) {
      total.push_back(*begin);
    } 

  }
  allCrossingPoints.clear();

  const vector<DetLayer*>& csclayers = theService->detLayerGeometry()->allCSCLayers();
  for (vector<DetLayer*>::const_iterator iLayer = csclayers.begin(); iLayer != csclayers.end(); ++iLayer) {

    GlobalPoint xPoint = crossingPoint(innerPos, outerPos, dynamic_cast<const ForwardDetLayer*>(*iLayer)); //crossing points of track with cylinder

    // check if point is inside the detector
    if ((fabs(xPoint.y()) < 1000.0) && (fabs(xPoint.z()) < 1500.0) 
           && (!(xPoint.y() == 0 && xPoint.x() == 0 && xPoint.z() == 0))) allCrossingPoints.push_back(xPoint);
   }
   stable_sort(allCrossingPoints.begin(), allCrossingPoints.end(), LessMag(innerPos) );

   vector<const GeomDet*> tempCSC;
   for (vector<GlobalPoint>::const_iterator ipos = allCrossingPoints.begin(); ipos != allCrossingPoints.end(); ++ipos) {

     tempCSC = cscPositionToDets(*ipos);
     vector<const GeomDet*>::const_iterator begin = tempCSC.begin();
     vector<const GeomDet*>::const_iterator end = tempCSC.end();

     for (; begin!=end;++begin) {
       total.push_back(*begin);
     }

  }

  return total;

}


//
// Intersection point of track with barrel layer
//
GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1, 
                                            const GlobalPoint& p2,
                                            const BarrelDetLayer* dl) const {

  const BoundCylinder& bc = dl->specificSurface();
  return crossingPoint(p1, p2, bc);

}


//
// Intersection point of track with cylinder
//
GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1,
                                            const GlobalPoint& p2, 
                                            const Cylinder& cyl) const {

  float radius = cyl.radius();

  GlobalVector dp = p1 - p2;
  float slope = dp.x()/dp.y();
  float a = p1.x() - slope * p1.y();

  float n2 = (1 + slope * slope);
  float n1 = 2*a*slope;
  float n0 = a*a - radius*radius;

  float y1 = 9999;
  float y2 = 9999;
  if ( n1*n1 - 4*n2*n0 > 0 ) {
    y1 = (-n1 + sqrt(n1*n1 - 4*n2*n0) ) / (2 * n2);
    y2 = (-n1 - sqrt(n1*n1 - 4*n2*n0) ) / (2 * n2);
  }

  float x1 = p1.x() + slope * (y1 - p1.y());
  float x2 = p1.x() + slope * (y2 - p1.y());

  float slopeZ = dp.z()/dp.y();

  float z1 = p1.z() + slopeZ * (y1 - p1.y());
  float z2 = p1.z() + slopeZ * (y2 - p1.y());

  // there are two crossing points, return the one that is in the same quadrant as point of extrapolation
  if ((p2.x()*x1 > 0) && (y1*p2.y() > 0) && (z1*p2.z() > 0)) {
    return GlobalPoint(x1, y1, z1); 
  } else { 
    return GlobalPoint(x2, y2, z2);
  }

}


//
// Intersection point of track with endcap disk
//
GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1, 
                                            const GlobalPoint& p2, 
                                            const ForwardDetLayer* dl) const {

  const BoundDisk& bc = dl->specificSurface();
  return crossingPoint(p1, p2, bc);
   
}  


//
// Intersection point of track with disk
//
GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1, 
                                            const GlobalPoint& p2, 
                                            const BoundDisk& disk) const {

  float diskZ = disk.position().z();
  int endcap =  diskZ > 0 ? 1 : (diskZ < 0 ? -1 : 0);
  diskZ = diskZ + endcap*dynamic_cast<const SimpleDiskBounds&>(disk.bounds()).thickness()/2.;

  std::cout << "compare z's: " << diskZ  << std::endl;

  // line connection innermost and outermost state on vector
  GlobalVector dp = p1 - p2;

  float slopeZ = dp.z()/dp.y();
  float y1 = diskZ / slopeZ;

  float slopeX = dp.z()/dp.x();
  float x1 = diskZ / slopeX;

  float z1 = diskZ;

  if (p2.z()*z1 > 0) {
    return GlobalPoint(x1, y1, z1);
  } else {
    return GlobalPoint(0, 0, 0);
  }

}


//
// GeomDets along the track in DT
//
vector<const GeomDet*> MuonShowerInformationFiller::dtPositionToDets(const GlobalPoint& gp) const {

   int minwheel = -3;
   int maxwheel = -3;
   if ( gp.z() < -680.0 ) { minwheel = -3; maxwheel = -3;}
   else if ( gp.z() < -396.0 ) { minwheel = -2; maxwheel = -1;}
   else if (gp.z() < -126.8) { minwheel = -2; maxwheel = 0; }
   else if (gp.z() < 126.8) { minwheel = -1; maxwheel = 1; }
   else if (gp.z() < 396.0) { minwheel = 0; maxwheel = 2; }
   else if (gp.z() < 680.0) { minwheel = 1; maxwheel = 2; }
   else { minwheel = 3; maxwheel = 3; }

   int station = 5;
   if ( gp.perp() > 680.0 && gp.perp() < 755.0 ) station = 4;
   else if ( gp.perp() > 580.0 ) station = 3;
   else if ( gp.perp() > 480.0 ) station = 2;
   else if ( gp.perp() > 380.0 ) station = 1;
   else station = 0;

   vector<int> sectors;

   float phistep = M_PI/6;

   float phigp = (float)gp.phi();

   if ( fabs(deltaPhi(phigp, 0*phistep)) < phistep ) sectors.push_back(1);
   if ( fabs(deltaPhi(phigp, phistep))   < phistep ) sectors.push_back(2);
   if ( fabs(deltaPhi(phigp, 2*phistep)) < phistep ) sectors.push_back(3);
   if ( fabs(deltaPhi(phigp, 3*phistep)) < phistep ) {
       sectors.push_back(4);
       if (station == 4) sectors.push_back(13);
   }
   if ( fabs(deltaPhi(phigp, 4*phistep)) < phistep ) sectors.push_back(5);
   if ( fabs(deltaPhi(phigp, 5*phistep)) < phistep ) sectors.push_back(6);
   if ( fabs(deltaPhi(phigp, 6*phistep)) < phistep ) sectors.push_back(7);
   if ( fabs(deltaPhi(phigp, 7*phistep)) < phistep ) sectors.push_back(8);
   if ( fabs(deltaPhi(phigp, 8*phistep)) < phistep ) sectors.push_back(9);
   if ( fabs(deltaPhi(phigp, 9*phistep)) < phistep ) {
       sectors.push_back(10);
       if (station == 4) sectors.push_back(14);
   }
   if ( fabs(deltaPhi(phigp, 10*phistep)) < phistep ) sectors.push_back(11);
   if ( fabs(deltaPhi(phigp, 11*phistep)) < phistep ) sectors.push_back(12);

   LogTrace(category_) << "DT position to dets" << endl;
   LogTrace(category_) << "number of sectors to consider: " << sectors.size() << endl;   
   LogTrace(category_) << "station: " << station << " wheels: " << minwheel << " " << maxwheel << endl;

   vector<const GeomDet*> result;
   if (station > 4 || station < 1) return result;
   if (minwheel > 2 || maxwheel < -2) return result;

   for (vector<int>::const_iterator isector = sectors.begin(); isector != sectors.end(); ++isector ) {
     for (int iwheel = minwheel; iwheel != maxwheel + 1; ++iwheel) {
       DTChamberId chamberid(iwheel, station, (*isector));
       result.push_back(theService->trackingGeometry()->idToDet(chamberid));
     }
   }

   LogTrace(category_) << "number of GeomDets for this track: " << result.size() << endl;

   return result;

}


//
// GeomDets along the track in CSC
//
vector<const GeomDet*> MuonShowerInformationFiller::cscPositionToDets(const GlobalPoint& gp) const {

  // determine the endcap side
  int endcap = 0;
  if (gp.z() > 0) {endcap = 1;} else {endcap = 2;} 

  // determine the csc station and range of rings
  int station = 5;

  // check all rings in a station
  if ( fabs(gp.z()) > 1000. && fabs(gp.z()) < 1055.0 ) {
    station = 4;
  }        
  else if ( fabs(gp.z()) > 910.0 && fabs(gp.z()) < 965.0) {
    station = 3;
  }
  else if ( fabs(gp.z()) > 800.0 && fabs(gp.z()) < 860.0) {
    station = 2;
  }
  else if ( fabs(gp.z()) > 570.0 && fabs(gp.z()) < 730.0) {
    station = 1;
  }

  vector<int> sectors;

  float phistep1 = M_PI/18.; //for all the rings except first rings for stations > 1
  float phistep2 = M_PI/9.;
  float phigp = (float)gp.phi();

  int ring = -1;

  // determine the ring
  if (station == 1) {

//FIX ME!!! 
      if (gp.perp() >  100 && gp.perp() < 270) ring = 1;
      else if (gp.perp() > 270 && gp.perp() < 450) ring = 2;
      else if (gp.perp() > 450 && gp.perp() < 695) ring = 3;
      else if (gp.perp() > 100 && gp.perp() < 270) ring = 4;

  } 
  else if (station == 2) {
      
      if (gp.perp() > 140 && gp.perp() < 350) ring = 1;
      else if (gp.perp() > 350 && gp.perp() < 700) ring = 2;

  }
  else if (station == 3) {

      if (gp.perp() > 160 && gp.perp() < 350) ring = 1;
      else if (gp.perp() > 350 && gp.perp() < 700) ring = 2;

  }
  else if (station == 4) {

      if (gp.perp() > 175 && gp.perp() < 350) ring = 1;
      else if (gp.perp() > 350 && gp.perp() < 700) ring = 2;

  }

  if (station > 1 && ring == 1) { 

   // we have 18 sectors in that case
   for (int i = 0; i < 18; i++) { 
      if ( fabs(deltaPhi(phigp, i*phistep2)) < phistep2 ) sectors.push_back(i+1);
    }

  } else {

   // we have 36 sectors in that case
   for (int i = 0; i < 36; i++) {
     if ( fabs(deltaPhi(phigp, i*phistep1)) < phistep1 ) sectors.push_back(i+1);
   }
}

   LogTrace(category_) << "CSC position to dets" << endl;
   LogTrace(category_) << "ring: " << ring << endl;
   LogTrace(category_) << "endcap: " << endcap << endl;
   LogTrace(category_) << "station: " << station << endl;
   LogTrace(category_) << "CSC number of sectors to consider: " << sectors.size() << endl;


 // check exceptional cases
   vector<const GeomDet*> result;
   if (station > 4 || station < 1) return result;
   if (endcap == 0) return result;
   if (ring == -1) return result;

   int minlayer = 1;
   int maxlayer = 6;

   for (vector<int>::const_iterator isector = sectors.begin(); isector != sectors.end(); ++isector) {
     for (int ilayer = minlayer; ilayer != maxlayer + 1; ++ ilayer) {
       CSCDetId cscid(endcap, station, ring, (*isector), ilayer);
       result.push_back(theService->trackingGeometry()->idToDet(cscid));
     }
   }

  return result;

}


//
// Calculate shower size per station
//
vector<double> MuonShowerInformationFiller::stationShowerRSizes(const reco::Muon& muon) const {

  // initialize
  vector<double> stationShowerRSize(4,0.0);

  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.globalTrack();
  else if ( muon.isStandAloneMuon() )    track = muon.outerTrack();
  else return stationShowerRSize;

  //fill hits
  vector<MuonRecHitContainer> muonRecHits = fillHitsByStation(*track);
  MuonTransientTrackingRecHit::MuonRecHitContainer muonRecHitsPhiTemp, muonRecHitsThetaTemp, muonRecHitsPhiBest, muonRecHitsThetaBest;

  LogTrace(category_) << "StationShowerSizes, we filled 4 vectors for this track, check size before clustering " 
       << muonRecHits[0].size() << " " 
       << muonRecHits[1].size() << " " 
       << muonRecHits[2].size() << " " 
       << muonRecHits[3].size() << endl;

  // send station hits to the clustering algorithm
  for ( int stat = 0; stat != 4; stat++ ) {
    if (muonRecHits[stat].size() > 1) {
      stable_sort(muonRecHits[stat].begin(), muonRecHits[stat].end(), LessPhi());

      float dphimax = 0;
      float dthetamax = 0;
      for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator iseed = muonRecHits[stat].begin(); iseed != muonRecHits[stat].end(); ++iseed) { 
          if (!(*iseed)->isValid()) continue;
          GlobalPoint refpoint = (*iseed)->globalPosition(); //starting from the one with smallest value of phi
          muonRecHitsPhiTemp.clear(); 
          muonRecHitsThetaTemp.clear();
          muonRecHitsPhiTemp = findPhiCluster(muonRecHits[stat], refpoint); //get clustered hits for this iseed
          muonRecHitsThetaTemp = findThetaCluster(muonRecHits[stat], refpoint);
     if (muonRecHitsPhiTemp.size() > 1) {
      float dphi = fabs(deltaPhi((float)muonRecHitsPhiTemp.back()->globalPosition().phi(), (float)muonRecHitsPhiTemp.front()->globalPosition().phi()));
      //if this cluster is bigger than previous, then we keep it
      //we want to identify the big dphi cluster for each station
      if (dphi > dphimax) {
          dphimax = dphi;
          muonRecHitsPhiBest = muonRecHitsPhiTemp;
            }
         } //at least two hits
     if (muonRecHitsThetaTemp.size() > 1) {
      float dtheta = fabs((float)muonRecHitsThetaTemp.back()->globalPosition().theta() - (float)muonRecHitsThetaTemp.front()->globalPosition().theta());
      if (dtheta > dthetamax) {
          dthetamax = dtheta;
          muonRecHitsThetaBest = muonRecHitsThetaTemp;
            }
         } //at least two hits
      }
     if (muonRecHitsThetaBest.size() > 1 && muonRecHitsPhiBest.size() > 1) 
      stationShowerRSize.at(stat) = sqrt(pow(muonRecHitsPhiBest.front()->globalPosition().phi()-muonRecHitsPhiBest.back()->globalPosition().phi(),2)+pow(muonRecHitsThetaBest.front()->globalPosition().theta()-muonRecHitsThetaBest.back()->globalPosition().theta(),2));
    }//not empty container
  }//loop over stations

  return stationShowerRSize;
}

//
//
//
vector<double> MuonShowerInformationFiller::stationShowerSizes(const reco::Muon& muon) const {

  // initialize
  vector<double> stationShowerSize(4,0.0);

  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.globalTrack();
  else if ( muon.isStandAloneMuon() )    track = muon.outerTrack();
  else return stationShowerSize;

  //fill hits
  vector<MuonRecHitContainer> muonRecHits = fillHitsByStation(*track);
  MuonTransientTrackingRecHit::MuonRecHitContainer muonRecHitsPhiTemp, muonRecHitsPhiBest;

  LogTrace(category_) << "StationShowerSizes, we filled 4 vectors for this track, check size before clustering "
       << muonRecHits[0].size() << " "
       << muonRecHits[1].size() << " "
       << muonRecHits[2].size() << " "
       << muonRecHits[3].size() << endl;

  // send station hits to the clustering algorithm
  for ( int stat = 0; stat != 4; stat++ ) {
    if (!muonRecHits[stat].empty()) {
      stable_sort(muonRecHits[stat].begin(), muonRecHits[stat].end(), LessPhi());

      float dphimax = 0;
      for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator iseed = muonRecHits[stat].begin(); iseed != muonRecHits[stat].end(); ++iseed) {
          if (!(*iseed)->isValid()) continue;
          GlobalPoint refpoint = (*iseed)->globalPosition(); //starting from the one with smallest value of phi
          muonRecHitsPhiTemp.clear();
          muonRecHitsPhiTemp = findPhiCluster(muonRecHits[stat], refpoint); //get clustered hits for this iseed
     if (muonRecHitsPhiTemp.size() > 1) {
         float dphi = fabs(deltaPhi((float)muonRecHitsPhiTemp.back()->globalPosition().phi(), (float)muonRecHitsPhiTemp.front()->globalPosition().phi()));
      if (dphi > dphimax) {
          dphimax = dphi;
          muonRecHitsPhiBest = muonRecHitsPhiTemp;
            }
         } //at least two hits
      }//loop over seeds 
      if (!muonRecHitsPhiBest.empty()) 
      muonRecHits[stat] = muonRecHitsPhiBest;
      stable_sort(muonRecHits[stat].begin(), muonRecHits[stat].end(), LessAbsMag());
      muonRecHits[stat].front();
      GlobalPoint refpoint = muonRecHits[stat].front()->globalPosition();
      LogTrace(category_) << "dphimax " << dphimax << " refpoint " << refpoint << endl;
      stationShowerSize[stat] = refpoint.mag() * dphimax;
      LogTrace(category_) << stationShowerSize[stat] << endl;
    }//not empty container
  }//loop over stations

  LogTrace(category_) << "Cluster sizes "
       << stationShowerSize.at(0) << " "
       << stationShowerSize.at(1) << " "
       << stationShowerSize.at(2) << " "
       << stationShowerSize.at(3) << endl;

  return stationShowerSize;
}

//
// Set station uncorrelated hits
//
vector<int> MuonShowerInformationFiller::stationUncorrelatedHits(const reco::Muon& muon) const {

  // initialize
  int totalsublayer[4] = {12,12,12,8};
  vector<int> stationUncorrHits(4,0);

  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.globalTrack();
  else if ( muon.isStandAloneMuon() )    track = muon.outerTrack();
  else return stationUncorrHits;

  // fill hits
  vector<MuonRecHitContainer> muonRecHits = fillHitsByStation(*track);

  // get the list of compatible dets
  vector<const GeomDet*> compatibleLayers = getCompatibleDets(*track);

  // get 4D rechits and split by station
  MuonRecHitContainer muSegments[4];
  
  for (vector<const GeomDet*>::const_iterator igd = compatibleLayers.begin(); igd != compatibleLayers.end(); igd++ ) {
  
    // get det id
    DetId geoId = (*igd)->geographicalId();

    // skip tracker hits
    if (geoId.det()!= DetId::Muon) continue;

    // DT 
    if ( geoId.subdetId() == MuonSubdetId::DT ) {
         
      DTChamberId did(geoId.rawId());
      int station = did.station();

      // split 1D and 4D rechits by station
      muSegments[station-1] = recHits4D(*igd, theDT4DRecSegments, theCSCSegments);
          
    } 
    else if (geoId.subdetId() == MuonSubdetId::CSC) {

      CSCDetId did(geoId.rawId());
      int station = did.station();

      // split 1D and 4D rechits by station
      muSegments[station-1] = recHits4D(*igd, theDT4DRecSegments, theCSCSegments);

    } 
    else if (geoId.subdetId() == MuonSubdetId::RPC) {
      LogTrace(category_) << "Wrong subdet id" << endl;
    } 

  }
  LogTrace(category_) << "StationUncorrelatedHits, we filled 4 vectors for this track, check their sizes " 
       << muSegments[0].size() << " "  
       << muSegments[1].size() << " " 
       << muSegments[2].size() << " " 
       << muSegments[3].size() << endl;

  int ncorrelated[4] = {0,0,0,0};

  // calculate number of uncorrelated hits
  for (int stat = 0; stat < 4; stat++) {
    ncorrelated[stat] = numberOfCorrelatedHits(muSegments[stat]);     
    if (ncorrelated[stat] > totalsublayer[stat]) ncorrelated[stat] = totalsublayer[stat];
    stationUncorrHits[stat] =  max(0, int(muonRecHits[stat].size()) - ncorrelated[stat]);
  }

  LogTrace(category_) << "Uncorrelated hits "
       << stationUncorrHits.at(0) << " " 
       << stationUncorrHits.at(1) << " " 
       << stationUncorrHits.at(2) << " " 
       << stationUncorrHits.at(3) << endl;

  return stationUncorrHits;

}

//
//
//
vector<MuonTransientTrackingRecHit::MuonRecHitContainer> MuonShowerInformationFiller::fillHitsByStation(const reco::Track& track) const {

  // split 1D rechits by station
  vector<MuonRecHitContainer> muonRecHits(4);
/*
  reco::TrackRef track;
  if ( muon.isGlobalMuon()  )            track = muon.globalTrack();
  else if ( muon.isStandAloneMuon() )    track = muon.outerTrack();
  else return muonRecHits;
*/
  // get vector of GeomDets compatible with a track
  vector<const GeomDet*> compatibleLayers = getCompatibleDets(track);

  // for special cases: CSC station 1
  MuonRecHitContainer tmpCSC1;
  bool dtOverlapToCheck = false;
  bool cscOverlapToCheck = false;

  for (vector<const GeomDet*>::const_iterator igd = compatibleLayers.begin(); igd != compatibleLayers.end(); igd++ )  {

    // get det id
    DetId geoId = (*igd)->geographicalId();  

    // skip tracker hits
    if (geoId.det()!= DetId::Muon) continue;

    // DT 
    if ( geoId.subdetId() == MuonSubdetId::DT ) {

      // get station
      DTChamberId detid(geoId.rawId());
      int station = detid.station();
      int wheel = detid.wheel();

      if (abs(wheel) == 2 && station != 4 &&  station != 1) dtOverlapToCheck = true;

      // loop over all superlayers of a DT chamber
      for (int isuperlayer = DTChamberId::minSuperLayerId; isuperlayer != DTChamberId::maxSuperLayerId + 1; ++isuperlayer) {
        // loop over all layers inside the superlayer
        for (int ilayer = DTChamberId::minLayerId; ilayer != DTChamberId::maxLayerId+1; ++ilayer) {
          DTLayerId lid(detid, isuperlayer, ilayer);
          DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);
          for (DTRecHitCollection::const_iterator rechit = dRecHits.first; rechit != dRecHits.second;++rechit) {
            vector<const TrackingRecHit*> subrechits = (*rechit).recHits();
            // loop over rechits and put it into the vectors corresponding to superlayers
            for (vector<const TrackingRecHit*>::iterator irechit = subrechits.begin(); irechit != subrechits.end(); ++irechit) {
                     muonRecHits.at(station-1).push_back(MuonTransientTrackingRecHit::specificBuild((&**igd),&**irechit));
                }
             }
          }
       }
    }
    else if (geoId.subdetId() == MuonSubdetId::CSC) {

      // get station
      CSCDetId did(geoId.rawId());
      int station = did.station();
      int ring = did.ring();

      if ((station == 1 && ring == 3) && dtOverlapToCheck) cscOverlapToCheck = true; 

      // split 1D rechits by station
      CSCRecHit2DCollection::range dRecHits = theCSCRecHits->get(did);
      for (CSCRecHit2DCollection::const_iterator rechit = dRecHits.first; rechit != dRecHits.second; ++rechit) {

        if (!cscOverlapToCheck) { 
           muonRecHits.at(station-1).push_back(MuonTransientTrackingRecHit::specificBuild((&**igd),&*rechit));
           } else {             
             tmpCSC1.push_back(MuonTransientTrackingRecHit::specificBuild((&**igd),&*rechit));

             //sort by perp, then insert to appropriate container
             MuonRecHitContainer temp = findPerpCluster(tmpCSC1);
             if (temp.empty()) continue; 

             float center;
             if (temp.size() > 1) {
               center = (temp.front()->globalPosition().perp() + temp.back()->globalPosition().perp())/2.;
             } else {
               center = temp.front()->globalPosition().perp();
  	     }
             temp.clear();

             if (center > 550.) {
                muonRecHits.at(2).insert(muonRecHits.at(2).end(),tmpCSC1.begin(),tmpCSC1.end());
              } else {
              muonRecHits.at(1).insert(muonRecHits.at(1).end(),tmpCSC1.begin(),tmpCSC1.end());
	  } 
         tmpCSC1.clear();
        }
      }
    } else if (geoId.subdetId() == MuonSubdetId::RPC) {
      LogTrace(category_) << "Wrong subdet id" << endl;
    }

  }

  return muonRecHits;

}
