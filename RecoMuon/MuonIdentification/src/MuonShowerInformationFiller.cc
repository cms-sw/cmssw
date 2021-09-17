/**
 *  \package: MuonIdentification
 *  \class: MuonShowerInformationFiller
 *  Description: class for muon shower identification
 *

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
#include "Geometry/Records/interface/MuonGeometryRecord.h"

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
MuonShowerInformationFiller::MuonShowerInformationFiller(const edm::ParameterSet& par, edm::ConsumesCollector& iC)
    : theService(nullptr),
      theDTRecHitLabel(par.getParameter<InputTag>("DTRecSegmentLabel")),
      theCSCRecHitLabel(par.getParameter<InputTag>("CSCRecSegmentLabel")),
      theCSCSegmentsLabel(par.getParameter<InputTag>("CSCSegmentLabel")),
      theDT4DRecSegmentLabel(par.getParameter<InputTag>("DT4DRecSegmentLabel")) {
  theDTRecHitToken = iC.consumes<DTRecHitCollection>(theDTRecHitLabel);
  theCSCRecHitToken = iC.consumes<CSCRecHit2DCollection>(theCSCRecHitLabel);
  theCSCSegmentsToken = iC.consumes<CSCSegmentCollection>(theCSCSegmentsLabel);
  theDT4DRecSegmentToken = iC.consumes<DTRecSegment4DCollection>(theDT4DRecSegmentLabel);

  edm::ParameterSet serviceParameters = par.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, edm::ConsumesCollector(iC));

  theTrackerRecHitBuilderName = par.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = par.getParameter<string>("MuonRecHitBuilder");

  theCacheId_TRH = 0;
  theCacheId_MT = 0;

  category_ = "MuonShowerInformationFiller";

  for (int istat = 0; istat < 4; istat++) {
    theStationShowerDeltaR.push_back(0.);
    theStationShowerTSize.push_back(0.);
    theAllStationHits.push_back(0);
    theCorrelatedStationHits.push_back(0);
  }
}

//
// Destructor
//
MuonShowerInformationFiller::~MuonShowerInformationFiller() {
  if (theService)
    delete theService;
}

//
//Fill the MuonShower struct
//
reco::MuonShower MuonShowerInformationFiller::fillShowerInformation(const reco::Muon& muon,
                                                                    const edm::Event& iEvent,
                                                                    const edm::EventSetup& iSetup) {
  reco::MuonShower returnShower;

  // Update the services
  theService->update(iSetup);
  setEvent(iEvent);
  setServices(theService->eventSetup());

  fillHitsByStation(muon);
  std::vector<int> nStationHits = theAllStationHits;
  std::vector<int> nStationCorrelatedHits = theCorrelatedStationHits;
  std::vector<float> stationShowerSizeT = theStationShowerTSize;
  std::vector<float> stationShowerDeltaR = theStationShowerDeltaR;

  returnShower.nStationHits = nStationHits;
  returnShower.nStationCorrelatedHits = nStationCorrelatedHits;
  returnShower.stationShowerSizeT = stationShowerSizeT;
  returnShower.stationShowerDeltaR = stationShowerDeltaR;

  return returnShower;
}

//
// Set Event
//
void MuonShowerInformationFiller::setEvent(const edm::Event& event) {
  // get all the necesary products
  event.getByToken(theDTRecHitToken, theDTRecHits);
  event.getByToken(theCSCRecHitToken, theCSCRecHits);
  event.getByToken(theCSCSegmentsToken, theCSCSegments);
  event.getByToken(theDT4DRecSegmentToken, theDT4DRecSegments);

  for (int istat = 0; istat < 4; istat++) {
    theStationShowerDeltaR.at(istat) = 0.;
    theStationShowerTSize.at(istat) = 0.;
    theAllStationHits.at(istat) = 0;
    theCorrelatedStationHits.at(istat) = 0;
  }
}

//
// Set services
//
void MuonShowerInformationFiller::setServices(const EventSetup& setup) {
  // DetLayer Geometry
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);
  setup.get<IdealMagneticFieldRecord>().get(theField);
  setup.get<TrackerRecoGeometryRecord>().get(theTracker);
  setup.get<MuonGeometryRecord>().get(theCSCGeometry);
  setup.get<MuonGeometryRecord>().get(theDTGeometry);

  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if (newCacheId_TRH != theCacheId_TRH) {
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName, theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName, theMuonRecHitBuilder);
  }
}

//
// Get hits owned by segments
//
TransientTrackingRecHit::ConstRecHitContainer MuonShowerInformationFiller::hitsFromSegments(
    const GeomDet* geomDet,
    edm::Handle<DTRecSegment4DCollection> dtSegments,
    edm::Handle<CSCSegmentCollection> cscSegments) const {
  MuonTransientTrackingRecHit::MuonRecHitContainer segments;

  DetId geoId = geomDet->geographicalId();

  if (geoId.subdetId() == MuonSubdetId::DT) {
    DTChamberId chamberId(geoId.rawId());

    // loop on segments 4D
    DTRecSegment4DCollection::id_iterator chamberIdIt;
    for (chamberIdIt = dtSegments->id_begin(); chamberIdIt != dtSegments->id_end(); ++chamberIdIt) {
      if (*chamberIdIt != chamberId)
        continue;

      // Get the range for the corresponding ChamberId
      DTRecSegment4DCollection::range range = dtSegments->get((*chamberIdIt));

      for (DTRecSegment4DCollection::const_iterator iseg = range.first; iseg != range.second; ++iseg) {
        if (iseg->dimension() != 4)
          continue;
        segments.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet, &*iseg));
      }
    }
  } else if (geoId.subdetId() == MuonSubdetId::CSC) {
    CSCDetId did(geoId.rawId());

    for (CSCSegmentCollection::id_iterator chamberId = cscSegments->id_begin(); chamberId != cscSegments->id_end();
         ++chamberId) {
      if ((*chamberId).chamber() != did.chamber())
        continue;

      // Get the range for the corresponding ChamberId
      CSCSegmentCollection::range range = cscSegments->get((*chamberId));

      for (CSCSegmentCollection::const_iterator iseg = range.first; iseg != range.second; ++iseg) {
        if (iseg->dimension() != 3)
          continue;
        segments.push_back(MuonTransientTrackingRecHit::specificBuild(geomDet, &*iseg));
      }
    }
  } else {
    LogTrace(category_) << "Segments are not built in RPCs" << endl;
  }

  TransientTrackingRecHit::ConstRecHitContainer allhitscorrelated;

  if (segments.empty())
    return allhitscorrelated;

  TransientTrackingRecHit::ConstRecHitPointer muonRecHit(segments.front());
  allhitscorrelated = MuonTransientTrackingRecHitBreaker::breakInSubRecHits(muonRecHit, 2);

  if (segments.size() == 1)
    return allhitscorrelated;

  for (MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator iseg = segments.begin() + 1;
       iseg != segments.end();
       ++iseg) {
    TransientTrackingRecHit::ConstRecHitPointer muonRecHit((*iseg));
    TransientTrackingRecHit::ConstRecHitContainer hits1 =
        MuonTransientTrackingRecHitBreaker::breakInSubRecHits(muonRecHit, 2);

    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit1 = hits1.begin(); ihit1 != hits1.end();
         ++ihit1) {
      bool usedbefore = false;
      //unused      DetId thisID = (*ihit1)->geographicalId();
      //LocalPoint lp1dinsegHit = (*ihit1)->localPosition();
      GlobalPoint gp1dinsegHit = (*ihit1)->globalPosition();

      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit2 = allhitscorrelated.begin();
           ihit2 != allhitscorrelated.end();
           ++ihit2) {
        //unused        DetId thisID2 = (*ihit2)->geographicalId();
        //LocalPoint lp1dinsegHit2 = (*ihit2)->localPosition();
        GlobalPoint gp1dinsegHit2 = (*ihit2)->globalPosition();

        if ((gp1dinsegHit2 - gp1dinsegHit).mag() < 1.0)
          usedbefore = true;
      }
      if (!usedbefore)
        allhitscorrelated.push_back(*ihit1);
    }
  }

  return allhitscorrelated;
}

//
// Find cluster
//

TransientTrackingRecHit::ConstRecHitContainer MuonShowerInformationFiller::findThetaCluster(
    TransientTrackingRecHit::ConstRecHitContainer& muonRecHits, const GlobalPoint& refpoint) const {
  if (muonRecHits.empty())
    return muonRecHits;

  //clustering step by theta
  float step = 0.05;
  TransientTrackingRecHit::ConstRecHitContainer result;

  stable_sort(muonRecHits.begin(), muonRecHits.end(), AbsLessDTheta(refpoint));

  for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator ihit = muonRecHits.begin();
       ihit != muonRecHits.end() - 1;
       ++ihit) {
    if (fabs((*(ihit + 1))->globalPosition().theta() - (*ihit)->globalPosition().theta()) < step) {
      result.push_back(*ihit);
    } else {
      break;
    }
  }

  return result;
}

//
//Used to treat overlap region
//
MuonTransientTrackingRecHit::MuonRecHitContainer MuonShowerInformationFiller::findPerpCluster(
    MuonTransientTrackingRecHit::MuonRecHitContainer& muonRecHits) const {
  if (muonRecHits.empty())
    return muonRecHits;

  stable_sort(muonRecHits.begin(), muonRecHits.end(), LessPerp());

  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator seedhit =
      min_element(muonRecHits.begin(), muonRecHits.end(), LessPerp());

  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ihigh = seedhit;
  MuonTransientTrackingRecHit::MuonRecHitContainer::const_iterator ilow = seedhit;

  float step = 0.1;
  while (ihigh != muonRecHits.end() - 1 &&
         (fabs((*(ihigh + 1))->globalPosition().perp() - (*ihigh)->globalPosition().perp()) < step)) {
    ihigh++;
  }
  while (ilow != muonRecHits.begin() &&
         (fabs((*ilow)->globalPosition().perp() - (*(ilow - 1))->globalPosition().perp()) < step)) {
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

  LogTrace(category_) << "Consider a track " << track.p() << " eta: " << track.eta() << " phi " << track.phi() << endl;

  TrajectoryStateOnSurface innerTsos = trajectoryStateTransform::innerStateOnSurface(
      track, *theService->trackingGeometry(), &*theService->magneticField());
  TrajectoryStateOnSurface outerTsos = trajectoryStateTransform::outerStateOnSurface(
      track, *theService->trackingGeometry(), &*theService->magneticField());

  GlobalPoint innerPos = innerTsos.globalPosition();
  GlobalPoint outerPos = outerTsos.globalPosition();

  vector<GlobalPoint> allCrossingPoints;

  const vector<const DetLayer*>& dtlayers = theService->detLayerGeometry()->allDTLayers();

  for (auto iLayer = dtlayers.begin(); iLayer != dtlayers.end(); ++iLayer) {
    // crossing points of track with cylinder
    GlobalPoint xPoint = crossingPoint(innerPos, outerPos, dynamic_cast<const BarrelDetLayer*>(*iLayer));

    // check if point is inside the detector
    if ((fabs(xPoint.y()) < 1000.0) && (fabs(xPoint.z()) < 1500) &&
        (!(xPoint.y() == 0 && xPoint.x() == 0 && xPoint.z() == 0)))
      allCrossingPoints.push_back(xPoint);
  }

  stable_sort(allCrossingPoints.begin(), allCrossingPoints.end(), LessMag(innerPos));

  vector<const GeomDet*> tempDT;

  for (vector<GlobalPoint>::const_iterator ipos = allCrossingPoints.begin(); ipos != allCrossingPoints.end(); ++ipos) {
    tempDT = dtPositionToDets(*ipos);
    vector<const GeomDet*>::const_iterator begin = tempDT.begin();
    vector<const GeomDet*>::const_iterator end = tempDT.end();

    for (; begin != end; ++begin) {
      total.push_back(*begin);
    }
  }
  allCrossingPoints.clear();

  const vector<const DetLayer*>& csclayers = theService->detLayerGeometry()->allCSCLayers();
  for (auto iLayer = csclayers.begin(); iLayer != csclayers.end(); ++iLayer) {
    GlobalPoint xPoint = crossingPoint(innerPos, outerPos, dynamic_cast<const ForwardDetLayer*>(*iLayer));

    // check if point is inside the detector
    if ((fabs(xPoint.y()) < 1000.0) && (fabs(xPoint.z()) < 1500.0) &&
        (!(xPoint.y() == 0 && xPoint.x() == 0 && xPoint.z() == 0)))
      allCrossingPoints.push_back(xPoint);
  }
  stable_sort(allCrossingPoints.begin(), allCrossingPoints.end(), LessMag(innerPos));

  vector<const GeomDet*> tempCSC;
  for (vector<GlobalPoint>::const_iterator ipos = allCrossingPoints.begin(); ipos != allCrossingPoints.end(); ++ipos) {
    tempCSC = cscPositionToDets(*ipos);
    vector<const GeomDet*>::const_iterator begin = tempCSC.begin();
    vector<const GeomDet*>::const_iterator end = tempCSC.end();

    for (; begin != end; ++begin) {
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

GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1,
                                                       const GlobalPoint& p2,
                                                       const Cylinder& cyl) const {
  float radius = cyl.radius();

  GlobalVector dp = p1 - p2;
  float slope = dp.x() / dp.y();
  float a = p1.x() - slope * p1.y();

  float n2 = (1 + slope * slope);
  float n1 = 2 * a * slope;
  float n0 = a * a - radius * radius;

  float y1 = 9999;
  float y2 = 9999;
  if (n1 * n1 - 4 * n2 * n0 > 0) {
    y1 = (-n1 + sqrt(n1 * n1 - 4 * n2 * n0)) / (2 * n2);
    y2 = (-n1 - sqrt(n1 * n1 - 4 * n2 * n0)) / (2 * n2);
  }

  float x1 = p1.x() + slope * (y1 - p1.y());
  float x2 = p1.x() + slope * (y2 - p1.y());

  float slopeZ = dp.z() / dp.y();

  float z1 = p1.z() + slopeZ * (y1 - p1.y());
  float z2 = p1.z() + slopeZ * (y2 - p1.y());

  // there are two crossing points, return the one that is in the same quadrant as point of extrapolation
  if ((p2.x() * x1 > 0) && (y1 * p2.y() > 0) && (z1 * p2.z() > 0)) {
    return GlobalPoint(x1, y1, z1);
  } else {
    return GlobalPoint(x2, y2, z2);
  }
}

//
// Intersection point of track with a forward layer
//
GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1,
                                                       const GlobalPoint& p2,
                                                       const ForwardDetLayer* dl) const {
  const BoundDisk& bc = dl->specificSurface();
  return crossingPoint(p1, p2, bc);
}

GlobalPoint MuonShowerInformationFiller::crossingPoint(const GlobalPoint& p1,
                                                       const GlobalPoint& p2,
                                                       const BoundDisk& disk) const {
  float diskZ = disk.position().z();
  int endcap = diskZ > 0 ? 1 : (diskZ < 0 ? -1 : 0);
  diskZ = diskZ + endcap * dynamic_cast<const SimpleDiskBounds&>(disk.bounds()).thickness() / 2.;

  GlobalVector dp = p1 - p2;

  float slopeZ = dp.z() / dp.y();
  float y1 = diskZ / slopeZ;

  float slopeX = dp.z() / dp.x();
  float x1 = diskZ / slopeX;

  float z1 = diskZ;

  if (p2.z() * z1 > 0) {
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
  if (gp.z() < -680.0) {
    minwheel = -3;
    maxwheel = -3;
  } else if (gp.z() < -396.0) {
    minwheel = -2;
    maxwheel = -1;
  } else if (gp.z() < -126.8) {
    minwheel = -2;
    maxwheel = 0;
  } else if (gp.z() < 126.8) {
    minwheel = -1;
    maxwheel = 1;
  } else if (gp.z() < 396.0) {
    minwheel = 0;
    maxwheel = 2;
  } else if (gp.z() < 680.0) {
    minwheel = 1;
    maxwheel = 2;
  } else {
    minwheel = 3;
    maxwheel = 3;
  }

  int station = 5;
  if (gp.perp() > 680.0 && gp.perp() < 755.0)
    station = 4;
  else if (gp.perp() > 580.0)
    station = 3;
  else if (gp.perp() > 480.0)
    station = 2;
  else if (gp.perp() > 380.0)
    station = 1;
  else
    station = 0;

  vector<int> sectors;

  float phistep = M_PI / 6;

  float phigp = (float)gp.barePhi();

  if (fabs(deltaPhi(phigp, 0 * phistep)) < phistep)
    sectors.push_back(1);
  if (fabs(deltaPhi(phigp, phistep)) < phistep)
    sectors.push_back(2);
  if (fabs(deltaPhi(phigp, 2 * phistep)) < phistep)
    sectors.push_back(3);
  if (fabs(deltaPhi(phigp, 3 * phistep)) < phistep) {
    sectors.push_back(4);
    if (station == 4)
      sectors.push_back(13);
  }
  if (fabs(deltaPhi(phigp, 4 * phistep)) < phistep)
    sectors.push_back(5);
  if (fabs(deltaPhi(phigp, 5 * phistep)) < phistep)
    sectors.push_back(6);
  if (fabs(deltaPhi(phigp, 6 * phistep)) < phistep)
    sectors.push_back(7);
  if (fabs(deltaPhi(phigp, 7 * phistep)) < phistep)
    sectors.push_back(8);
  if (fabs(deltaPhi(phigp, 8 * phistep)) < phistep)
    sectors.push_back(9);
  if (fabs(deltaPhi(phigp, 9 * phistep)) < phistep) {
    sectors.push_back(10);
    if (station == 4)
      sectors.push_back(14);
  }
  if (fabs(deltaPhi(phigp, 10 * phistep)) < phistep)
    sectors.push_back(11);
  if (fabs(deltaPhi(phigp, 11 * phistep)) < phistep)
    sectors.push_back(12);

  LogTrace(category_) << "DT position to dets" << endl;
  LogTrace(category_) << "number of sectors to consider: " << sectors.size() << endl;
  LogTrace(category_) << "station: " << station << " wheels: " << minwheel << " " << maxwheel << endl;

  vector<const GeomDet*> result;
  if (station > 4 || station < 1)
    return result;
  if (minwheel > 2 || maxwheel < -2)
    return result;

  for (vector<int>::const_iterator isector = sectors.begin(); isector != sectors.end(); ++isector) {
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
  if (gp.z() > 0) {
    endcap = 1;
  } else {
    endcap = 2;
  }

  // determine the csc station and range of rings
  int station = 5;

  // check all rings in a station
  if (fabs(gp.z()) > 1000. && fabs(gp.z()) < 1055.0) {
    station = 4;
  } else if (fabs(gp.z()) > 910.0 && fabs(gp.z()) < 965.0) {
    station = 3;
  } else if (fabs(gp.z()) > 800.0 && fabs(gp.z()) < 860.0) {
    station = 2;
  } else if (fabs(gp.z()) > 570.0 && fabs(gp.z()) < 730.0) {
    station = 1;
  }

  vector<int> sectors;

  float phistep1 = M_PI / 18.;  //for all the rings except first rings for stations > 1
  float phistep2 = M_PI / 9.;
  float phigp = (float)gp.barePhi();

  int ring = -1;

  // determine the ring
  if (station == 1) {
    //FIX ME!!!
    if (gp.perp() > 100 && gp.perp() < 270)
      ring = 1;
    else if (gp.perp() > 270 && gp.perp() < 450)
      ring = 2;
    else if (gp.perp() > 450 && gp.perp() < 695)
      ring = 3;
    else if (gp.perp() > 100 && gp.perp() < 270)
      ring = 4;

  } else if (station == 2) {
    if (gp.perp() > 140 && gp.perp() < 350)
      ring = 1;
    else if (gp.perp() > 350 && gp.perp() < 700)
      ring = 2;

  } else if (station == 3) {
    if (gp.perp() > 160 && gp.perp() < 350)
      ring = 1;
    else if (gp.perp() > 350 && gp.perp() < 700)
      ring = 2;

  } else if (station == 4) {
    if (gp.perp() > 175 && gp.perp() < 350)
      ring = 1;
    else if (gp.perp() > 350 && gp.perp() < 700)
      ring = 2;
  }

  if (station > 1 && ring == 1) {
    // we have 18 sectors in that case
    for (int i = 0; i < 18; i++) {
      if (fabs(deltaPhi(phigp, i * phistep2)) < phistep2)
        sectors.push_back(i + 1);
    }

  } else {
    // we have 36 sectors in that case
    for (int i = 0; i < 36; i++) {
      if (fabs(deltaPhi(phigp, i * phistep1)) < phistep1)
        sectors.push_back(i + 1);
    }
  }

  LogTrace(category_) << "CSC position to dets" << endl;
  LogTrace(category_) << "ring: " << ring << endl;
  LogTrace(category_) << "endcap: " << endcap << endl;
  LogTrace(category_) << "station: " << station << endl;
  LogTrace(category_) << "CSC number of sectors to consider: " << sectors.size() << endl;

  // check exceptional cases
  vector<const GeomDet*> result;
  if (station > 4 || station < 1)
    return result;
  if (endcap == 0)
    return result;
  if (ring == -1)
    return result;

  int minlayer = 1;
  int maxlayer = 6;

  for (vector<int>::const_iterator isector = sectors.begin(); isector != sectors.end(); ++isector) {
    for (int ilayer = minlayer; ilayer != maxlayer + 1; ++ilayer) {
      CSCDetId cscid(endcap, station, ring, (*isector), ilayer);
      result.push_back(theService->trackingGeometry()->idToDet(cscid));
    }
  }

  return result;
}

//
//Set class members
//
void MuonShowerInformationFiller::fillHitsByStation(const reco::Muon& muon) {
  reco::TrackRef track;
  if (muon.isGlobalMuon())
    track = muon.globalTrack();
  else if (muon.isStandAloneMuon())
    track = muon.outerTrack();
  else
    return;

  // split 1D rechits by station
  vector<MuonRecHitContainer> muonRecHits(4);

  // split rechits from segs by station
  vector<TransientTrackingRecHit::ConstRecHitContainer> muonCorrelatedHits(4);

  // get vector of GeomDets compatible with a track
  vector<const GeomDet*> compatibleLayers = getCompatibleDets(*track);

  // for special cases: CSC station 1
  MuonRecHitContainer tmpCSC1;
  bool dtOverlapToCheck = false;
  bool cscOverlapToCheck = false;

  for (vector<const GeomDet*>::const_iterator igd = compatibleLayers.begin(); igd != compatibleLayers.end(); igd++) {
    // get det id
    DetId geoId = (*igd)->geographicalId();

    // skip tracker hits
    if (geoId.det() != DetId::Muon)
      continue;

    // DT
    if (geoId.subdetId() == MuonSubdetId::DT) {
      // get station
      DTChamberId detid(geoId.rawId());
      int station = detid.station();
      int wheel = detid.wheel();

      // get rechits from segments per station
      TransientTrackingRecHit::ConstRecHitContainer muonCorrelatedHitsTmp =
          hitsFromSegments(*igd, theDT4DRecSegments, theCSCSegments);
      TransientTrackingRecHit::ConstRecHitContainer::const_iterator hits_begin = muonCorrelatedHitsTmp.begin();
      TransientTrackingRecHit::ConstRecHitContainer::const_iterator hits_end = muonCorrelatedHitsTmp.end();

      for (; hits_begin != hits_end; ++hits_begin) {
        muonCorrelatedHits.at(station - 1).push_back(*hits_begin);
      }

      //check overlap certain wheels and stations
      if (abs(wheel) == 2 && station != 4 && station != 1)
        dtOverlapToCheck = true;

      // loop over all superlayers of a DT chamber
      for (int isuperlayer = DTChamberId::minSuperLayerId; isuperlayer != DTChamberId::maxSuperLayerId + 1;
           ++isuperlayer) {
        // loop over all layers inside the superlayer
        for (int ilayer = DTChamberId::minLayerId; ilayer != DTChamberId::maxLayerId + 1; ++ilayer) {
          DTLayerId lid(detid, isuperlayer, ilayer);
          DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);
          for (DTRecHitCollection::const_iterator rechit = dRecHits.first; rechit != dRecHits.second; ++rechit) {
            vector<const TrackingRecHit*> subrechits = (*rechit).recHits();
            for (vector<const TrackingRecHit*>::iterator irechit = subrechits.begin(); irechit != subrechits.end();
                 ++irechit) {
              muonRecHits.at(station - 1).push_back(MuonTransientTrackingRecHit::specificBuild((&**igd), &**irechit));
            }
          }
        }
      }
    } else if (geoId.subdetId() == MuonSubdetId::CSC) {
      // get station
      CSCDetId did(geoId.rawId());
      int station = did.station();
      int ring = did.ring();

      //get rechits from segments by station
      TransientTrackingRecHit::ConstRecHitContainer muonCorrelatedHitsTmp =
          hitsFromSegments(*igd, theDT4DRecSegments, theCSCSegments);
      TransientTrackingRecHit::ConstRecHitContainer::const_iterator hits_begin = muonCorrelatedHitsTmp.begin();
      TransientTrackingRecHit::ConstRecHitContainer::const_iterator hits_end = muonCorrelatedHitsTmp.end();

      for (; hits_begin != hits_end; ++hits_begin) {
        muonCorrelatedHits.at(station - 1).push_back(*hits_begin);
      }

      if ((station == 1 && ring == 3) && dtOverlapToCheck)
        cscOverlapToCheck = true;

      // split 1D rechits by station
      CSCRecHit2DCollection::range dRecHits = theCSCRecHits->get(did);
      for (CSCRecHit2DCollection::const_iterator rechit = dRecHits.first; rechit != dRecHits.second; ++rechit) {
        if (!cscOverlapToCheck) {
          muonRecHits.at(station - 1).push_back(MuonTransientTrackingRecHit::specificBuild((&**igd), &*rechit));
        } else {
          tmpCSC1.push_back(MuonTransientTrackingRecHit::specificBuild((&**igd), &*rechit));

          //sort by perp, then insert to appropriate container
          MuonRecHitContainer temp = findPerpCluster(tmpCSC1);
          if (temp.empty())
            continue;

          float center;
          if (temp.size() > 1) {
            center = (temp.front()->globalPosition().perp() + temp.back()->globalPosition().perp()) / 2.;
          } else {
            center = temp.front()->globalPosition().perp();
          }
          temp.clear();

          if (center > 550.) {
            muonRecHits.at(2).insert(muonRecHits.at(2).end(), tmpCSC1.begin(), tmpCSC1.end());
          } else {
            muonRecHits.at(1).insert(muonRecHits.at(1).end(), tmpCSC1.begin(), tmpCSC1.end());
          }
          tmpCSC1.clear();
        }
      }
    } else if (geoId.subdetId() == MuonSubdetId::RPC) {
      LogTrace(category_) << "Wrong subdet id" << endl;
    }
  }  //loop over GeomDets compatible with a track

  // calculate number of all and correlated hits
  for (int stat = 0; stat < 4; stat++) {
    theCorrelatedStationHits[stat] = muonCorrelatedHits.at(stat).size();
    theAllStationHits[stat] = muonRecHits[stat].size();
  }
  LogTrace(category_) << "Hits used by the segments, by station " << theCorrelatedStationHits.at(0) << " "
                      << theCorrelatedStationHits.at(1) << " " << theCorrelatedStationHits.at(2) << " "
                      << theCorrelatedStationHits.at(3) << endl;

  LogTrace(category_) << "All DT 1D/CSC 2D  hits, by station " << theAllStationHits.at(0) << " "
                      << theAllStationHits.at(1) << " " << theAllStationHits.at(2) << " " << theAllStationHits.at(3)
                      << endl;

  //station shower sizes
  MuonTransientTrackingRecHit::MuonRecHitContainer muonRecHitsPhiBest;
  TransientTrackingRecHit::ConstRecHitContainer muonRecHitsThetaTemp, muonRecHitsThetaBest;
  // send station hits to the clustering algorithm
  for (int stat = 0; stat != 4; stat++) {
    const size_t nhit = muonRecHits[stat].size();
    if (nhit < 2)
      continue;  // Require at least 2 hits
    muonRecHitsPhiBest.clear();
    muonRecHitsPhiBest.reserve(nhit);

    // Cluster seeds by global position phi. Best cluster is chosen to give greatest dphi
    // Sort by phi (complexity = NLogN with enough memory, or = NLog^2N for insufficient mem)
    stable_sort(muonRecHits[stat].begin(), muonRecHits[stat].end(), LessPhi());

    // Search for gaps (complexity = N)
    std::vector<size_t> clUppers;
    for (size_t ihit = 0; ihit < nhit; ++ihit) {
      const size_t jhit = (ihit + 1) % nhit;
      const double phi1 = muonRecHits[stat].at(ihit)->globalPosition().barePhi();
      const double phi2 = muonRecHits[stat].at(jhit)->globalPosition().barePhi();

      const double dphi = std::abs(deltaPhi(phi1, phi2));
      if (dphi >= 0.05)
        clUppers.push_back(ihit);
    }

    //station shower sizes
    double dphimax = 0;
    if (clUppers.empty()) {
      // No gaps - there is only one cluster. Take all of them
      const double refPhi = muonRecHits[stat].at(0)->globalPosition().barePhi();
      double dphilo = 0, dphihi = 0;
      for (auto& hit : muonRecHits[stat]) {
        muonRecHitsPhiBest.push_back(hit);
        const double phi = hit->globalPosition().barePhi();
        dphilo = std::min(dphilo, deltaPhi(refPhi, phi));
        dphihi = std::max(dphihi, deltaPhi(refPhi, phi));
      }
      dphimax = std::abs(dphihi + dphilo);
    } else {
      // Loop over gaps to find the one with maximum dphi(begin, end)
      // By construction, number of gap must be greater than 1.
      size_t bestUpper = 0, bestLower = 0;
      for (auto icl = clUppers.begin(); icl != clUppers.end(); ++icl) {
        // upper bound of this cluster
        const size_t upper = *icl;
        // lower bound is +1 of preceeding upper bound
        const auto prevCl = (icl == clUppers.begin()) ? clUppers.end() : icl;
        const size_t lower = (*(prevCl - 1) + 1) % nhit;

        // At least two hit for a cluster
        if (upper == lower)
          continue;

        const double phi1 = muonRecHits[stat].at(upper)->globalPosition().barePhi();
        const double phi2 = muonRecHits[stat].at(lower)->globalPosition().barePhi();

        const double dphi = std::abs(deltaPhi(phi1, phi2));
        if (dphimax < dphi) {
          dphimax = dphi;
          bestUpper = upper;
          bestLower = lower;
        }
      }

      if (bestUpper > bestLower) {
        muonRecHitsPhiBest.reserve(bestUpper - bestLower + 1);
        std::copy(muonRecHits[stat].begin() + bestLower,
                  muonRecHits[stat].begin() + bestUpper + 1,
                  std::back_inserter(muonRecHitsPhiBest));
      } else if (bestUpper < bestLower) {
        muonRecHitsPhiBest.reserve(bestUpper + (nhit - bestLower) + 1);
        std::copy(muonRecHits[stat].begin(),
                  muonRecHits[stat].begin() + bestUpper + 1,
                  std::back_inserter(muonRecHitsPhiBest));
        std::copy(
            muonRecHits[stat].begin() + bestLower, muonRecHits[stat].end(), std::back_inserter(muonRecHitsPhiBest));
      }
    }

    //fill showerTs
    if (!muonRecHitsPhiBest.empty()) {
      muonRecHits[stat] = muonRecHitsPhiBest;
      stable_sort(muonRecHits[stat].begin(), muonRecHits[stat].end(), LessAbsMag());
      muonRecHits[stat].front();
      GlobalPoint refpoint = muonRecHits[stat].front()->globalPosition();
      theStationShowerTSize.at(stat) = refpoint.mag() * dphimax;
    }

    //for theta
    if (!muonCorrelatedHits.at(stat).empty()) {
      float dthetamax = 0;
      for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator iseed = muonCorrelatedHits.at(stat).begin();
           iseed != muonCorrelatedHits.at(stat).end();
           ++iseed) {
        if (!(*iseed)->isValid())
          continue;
        GlobalPoint refpoint = (*iseed)->globalPosition();  //starting from the one with smallest value of phi
        muonRecHitsThetaTemp.clear();
        muonRecHitsThetaTemp = findThetaCluster(muonCorrelatedHits.at(stat), refpoint);
        if (muonRecHitsThetaTemp.size() > 1) {
          float dtheta = fabs((float)muonRecHitsThetaTemp.back()->globalPosition().theta() -
                              (float)muonRecHitsThetaTemp.front()->globalPosition().theta());
          if (dtheta > dthetamax) {
            dthetamax = dtheta;
            muonRecHitsThetaBest = muonRecHitsThetaTemp;
          }
        }  //at least two hits
      }    //loop over seeds
    }      //not empty container2

    //fill deltaRs
    if (muonRecHitsThetaBest.size() > 1 && muonRecHitsPhiBest.size() > 1)
      theStationShowerDeltaR.at(stat) = sqrt(pow(deltaPhi(muonRecHitsPhiBest.front()->globalPosition().barePhi(),
                                                          muonRecHitsPhiBest.back()->globalPosition().barePhi()),
                                                 2) +
                                             pow(muonRecHitsThetaBest.front()->globalPosition().theta() -
                                                     muonRecHitsThetaBest.back()->globalPosition().theta(),
                                                 2));

  }  //loop over station

  LogTrace(category_) << "deltaR around a track containing all the station hits, by station "
                      << theStationShowerDeltaR.at(0) << " " << theStationShowerDeltaR.at(1) << " "
                      << theStationShowerDeltaR.at(2) << " " << theStationShowerDeltaR.at(3) << endl;

  LogTrace(category_) << "Transverse cluster size, by station " << theStationShowerTSize.at(0) << " "
                      << theStationShowerTSize.at(1) << " " << theStationShowerTSize.at(2) << " "
                      << theStationShowerTSize.at(3) << endl;

  return;
}
