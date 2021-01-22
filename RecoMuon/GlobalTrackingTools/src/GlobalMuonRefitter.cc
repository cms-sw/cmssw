/**
 *  Class: GlobalMuonRefitter
 *
 *  Description:
 *
 *
 *
 *  Authors :
 *  P. Traczyk, SINS Warsaw
 *
 *  \modified by C. Calabria, INFN & Universita Bari
 *  \modified by D. Nash, Northeastern University
 *  \modified by C. Caputo, UCLouvain
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "TrackingTools/TrackFitters/interface/RecHitLessByDet.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/DynamicTruncation.h"

using namespace std;
using namespace edm;

//----------------
// Constructors --
//----------------

GlobalMuonRefitter::GlobalMuonRefitter(const edm::ParameterSet& par,
                                       const MuonServiceProxy* service,
                                       edm::ConsumesCollector& iC)
    : theCosmicFlag(par.getParameter<bool>("PropDirForCosmics")),
      theDTRecHitLabel(par.getParameter<InputTag>("DTRecSegmentLabel")),
      theCSCRecHitLabel(par.getParameter<InputTag>("CSCRecSegmentLabel")),
      theGEMRecHitLabel(par.getParameter<InputTag>("GEMRecHitLabel")),
      theME0RecHitLabel(par.getParameter<InputTag>("ME0RecHitLabel")),
      theService(service) {
  theCategory = par.getUntrackedParameter<string>("Category", "Muon|RecoMuon|GlobalMuon|GlobalMuonRefitter");

  theHitThreshold = par.getParameter<int>("HitThreshold");
  theDTChi2Cut = par.getParameter<double>("Chi2CutDT");
  theCSCChi2Cut = par.getParameter<double>("Chi2CutCSC");
  theRPCChi2Cut = par.getParameter<double>("Chi2CutRPC");
  theGEMChi2Cut = par.getParameter<double>("Chi2CutGEM");
  theME0Chi2Cut = par.getParameter<double>("Chi2CutME0");

  // Refit direction
  string refitDirectionName = par.getParameter<string>("RefitDirection");

  if (refitDirectionName == "insideOut")
    theRefitDirection = insideOut;
  else if (refitDirectionName == "outsideIn")
    theRefitDirection = outsideIn;
  else
    throw cms::Exception("TrackTransformer constructor")
        << "Wrong refit direction chosen in TrackTransformer ParameterSet"
        << "\n"
        << "Possible choices are:"
        << "\n"
        << "RefitDirection = insideOut or RefitDirection = outsideIn";

  theFitterToken = iC.esConsumes(edm::ESInputTag("", par.getParameter<string>("Fitter")));
  thePropagatorName = par.getParameter<string>("Propagator");

  theSkipStation = par.getParameter<int>("SkipStation");
  theTrackerSkipSystem = par.getParameter<int>("TrackerSkipSystem");
  theTrackerSkipSection = par.getParameter<int>("TrackerSkipSection");  //layer, wheel, or disk depending on the system

  theTrackerRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", par.getParameter<string>("TrackerRecHitBuilder")));
  theMuonRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", par.getParameter<string>("MuonRecHitBuilder")));

  theRPCInTheFit = par.getParameter<bool>("RefitRPCHits");

  theDYTthrs = par.getParameter<std::vector<int> >("DYTthrs");
  theDYTselector = par.getParameter<int>("DYTselector");
  theDYTupdator = par.getParameter<bool>("DYTupdator");
  theDYTuseAPE = par.getParameter<bool>("DYTuseAPE");
  theDYTParThrsMode = par.getParameter<bool>("DYTuseThrsParametrization");
  if (theDYTParThrsMode)
    theDYTthrsParameters = par.getParameter<edm::ParameterSet>("DYTthrsParameters");
  dytInfo = new reco::DYTInfo();

  if (par.existsAs<double>("RescaleErrorFactor")) {
    theRescaleErrorFactor = par.getParameter<double>("RescaleErrorFactor");
    edm::LogWarning("GlobalMuonRefitter") << "using error rescale factor " << theRescaleErrorFactor;
  } else
    theRescaleErrorFactor = 1000.;

  theCacheId_TRH = 0;
  theDTRecHitToken = iC.consumes<DTRecHitCollection>(theDTRecHitLabel);
  theCSCRecHitToken = iC.consumes<CSCRecHit2DCollection>(theCSCRecHitLabel);
  theGEMRecHitToken = iC.consumes<GEMRecHitCollection>(theGEMRecHitLabel);
  theME0RecHitToken = iC.consumes<ME0SegmentCollection>(theME0RecHitLabel);
  CSCSegmentsToken = iC.consumes<CSCSegmentCollection>(InputTag("cscSegments"));
  all4DSegmentsToken = iC.consumes<DTRecSegment4DCollection>(InputTag("dt4DSegments"));
}

//--------------
// Destructor --
//--------------

GlobalMuonRefitter::~GlobalMuonRefitter() { delete dytInfo; }

//
// set Event
//
void GlobalMuonRefitter::setEvent(const edm::Event& event) {
  theEvent = &event;
  event.getByToken(theDTRecHitToken, theDTRecHits);
  event.getByToken(theCSCRecHitToken, theCSCRecHits);
  event.getByToken(theGEMRecHitToken, theGEMRecHits);
  event.getByToken(theME0RecHitToken, theME0RecHits);
  event.getByToken(CSCSegmentsToken, CSCSegments);
  event.getByToken(all4DSegmentsToken, all4DSegments);
}

void GlobalMuonRefitter::setServices(const EventSetup& setup) {
  theFitter = setup.getData(theFitterToken).clone();

  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if (newCacheId_TRH != theCacheId_TRH) {
    LogDebug(theCategory) << "TransientRecHitRecord changed!";
    theTrackerRecHitBuilder = &setup.getData(theTrackerRecHitBuilderToken);
    theMuonRecHitBuilder = &setup.getData(theMuonRecHitBuilderToken);
    hitCloner = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder)->cloner();
  }
  theFitter->setHitCloner(&hitCloner);
}

//
// build a combined tracker-muon trajectory
//
vector<Trajectory> GlobalMuonRefitter::refit(const reco::Track& globalTrack,
                                             const int theMuonHitsOption,
                                             const TrackerTopology* tTopo) const {
  LogTrace(theCategory) << " *** GlobalMuonRefitter *** option " << theMuonHitsOption << endl;

  ConstRecHitContainer allRecHitsTemp;  // all muon rechits temp

  reco::TransientTrack track(globalTrack, &*(theService->magneticField()), theService->trackingGeometry());

  auto tkbuilder = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder);

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if ((*hit)->isValid()) {
      if ((*hit)->geographicalId().det() == DetId::Tracker)
        allRecHitsTemp.push_back((**hit).cloneForFit(*tkbuilder->geometry()->idToDet((**hit).geographicalId())));
      else if ((*hit)->geographicalId().det() == DetId::Muon) {
        if ((*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit) {
          LogTrace(theCategory) << "RPC Rec Hit discarged";
          continue;
        }
        allRecHitsTemp.push_back(theMuonRecHitBuilder->build(&**hit));
      }
    }
  vector<Trajectory> refitted = refit(globalTrack, track, allRecHitsTemp, theMuonHitsOption, tTopo);
  return refitted;
}

//
// build a combined tracker-muon trajectory
//
vector<Trajectory> GlobalMuonRefitter::refit(const reco::Track& globalTrack,
                                             const reco::TransientTrack track,
                                             const TransientTrackingRecHit::ConstRecHitContainer& allRecHitsTemp,
                                             const int theMuonHitsOption,
                                             const TrackerTopology* tTopo) const {
  // MuonHitsOption: 0 - tracker only
  //                 1 - include all muon hits
  //                 2 - include only first muon hit(s)
  //                 3 - include only selected muon hits
  //                 4 - redo pattern recognition with dynamic truncation

  vector<int> stationHits(4, 0);
  map<DetId, int> hitMap;

  ConstRecHitContainer allRecHits;       // all muon rechits
  ConstRecHitContainer fmsRecHits;       // only first muon rechits
  ConstRecHitContainer selectedRecHits;  // selected muon rechits
  ConstRecHitContainer DYTRecHits;       // rec hits from dynamic truncation algorithm

  LogTrace(theCategory) << " *** GlobalMuonRefitter *** option " << theMuonHitsOption << endl;
  LogTrace(theCategory) << " Track momentum before refit: " << globalTrack.pt() << endl;
  LogTrace(theCategory) << " Hits size before : " << allRecHitsTemp.size() << endl;

  allRecHits = getRidOfSelectStationHits(allRecHitsTemp, tTopo);
  //    printHits(allRecHits);
  LogTrace(theCategory) << " Hits size: " << allRecHits.size() << endl;

  vector<Trajectory> outputTraj;

  if ((theMuonHitsOption == 1) || (theMuonHitsOption == 3) || (theMuonHitsOption == 4)) {
    // refit the full track with all muon hits
    vector<Trajectory> globalTraj = transform(globalTrack, track, allRecHits);

    if (globalTraj.empty()) {
      LogTrace(theCategory) << "No trajectory from the TrackTransformer!" << endl;
      return vector<Trajectory>();
    }

    LogTrace(theCategory) << " Initial trajectory state: "
                          << globalTraj.front().lastMeasurement().updatedState().freeState()->parameters() << endl;

    if (theMuonHitsOption == 1)
      outputTraj.push_back(globalTraj.front());

    if (theMuonHitsOption == 3) {
      checkMuonHits(globalTrack, allRecHits, hitMap);
      selectedRecHits = selectMuonHits(globalTraj.front(), hitMap);
      LogTrace(theCategory) << " Selected hits size: " << selectedRecHits.size() << endl;
      outputTraj = transform(globalTrack, track, selectedRecHits);
    }

    if (theMuonHitsOption == 4) {
      //
      // DYT 2.0
      //
      DynamicTruncation dytRefit(*theEvent, *theService);
      dytRefit.setProd(all4DSegments, CSCSegments);
      dytRefit.setSelector(theDYTselector);
      dytRefit.setThr(theDYTthrs);
      dytRefit.setUpdateState(theDYTupdator);
      dytRefit.setUseAPE(theDYTuseAPE);
      if (theDYTParThrsMode) {
        dytRefit.setParThrsMode(theDYTParThrsMode);
        dytRefit.setThrsMap(theDYTthrsParameters);
        dytRefit.setRecoP(globalTrack.p());
        dytRefit.setRecoEta(globalTrack.eta());
      }
      DYTRecHits = dytRefit.filter(globalTraj.front());
      dytInfo->CopyFrom(dytRefit.getDYTInfo());
      if ((DYTRecHits.size() > 1) &&
          (DYTRecHits.front()->globalPosition().mag() > DYTRecHits.back()->globalPosition().mag()))
        stable_sort(DYTRecHits.begin(), DYTRecHits.end(), RecHitLessByDet(alongMomentum));
      outputTraj = transform(globalTrack, track, DYTRecHits);
    }

  } else if (theMuonHitsOption == 2) {
    getFirstHits(globalTrack, allRecHits, fmsRecHits);
    outputTraj = transform(globalTrack, track, fmsRecHits);
  }

  if (!outputTraj.empty()) {
    LogTrace(theCategory) << "Refitted pt: "
                          << outputTraj.front().firstMeasurement().updatedState().globalParameters().momentum().perp()
                          << endl;
    return outputTraj;
  } else {
    LogTrace(theCategory) << "No refitted Tracks... " << endl;
    return vector<Trajectory>();
  }
}

//
//
//
void GlobalMuonRefitter::checkMuonHits(const reco::Track& muon,
                                       ConstRecHitContainer& all,
                                       map<DetId, int>& hitMap) const {
  LogTrace(theCategory) << " GlobalMuonRefitter::checkMuonHits " << endl;

  float coneSize = 20.0;

  // loop through all muon hits and calculate the maximum # of hits in each chamber
  for (ConstRecHitContainer::const_iterator imrh = all.begin(); imrh != all.end(); imrh++) {
    if ((*imrh != nullptr) && !(*imrh)->isValid())
      continue;

    int detRecHits = 0;
    MuonRecHitContainer dRecHits;

    DetId id = (*imrh)->geographicalId();
    DetId chamberId;

    // Skip tracker hits
    if (id.det() != DetId::Muon)
      continue;

    if (id.subdetId() == MuonSubdetId::DT) {
      DTChamberId did(id.rawId());
      chamberId = did;

      if ((*imrh)->dimension() > 1) {
        std::vector<const TrackingRecHit*> hits2d = (*imrh)->recHits();
        for (std::vector<const TrackingRecHit*>::const_iterator hit2d = hits2d.begin(); hit2d != hits2d.end();
             hit2d++) {
          if ((*hit2d)->dimension() > 1) {
            std::vector<const TrackingRecHit*> hits1d = (*hit2d)->recHits();
            for (std::vector<const TrackingRecHit*>::const_iterator hit1d = hits1d.begin(); hit1d != hits1d.end();
                 hit1d++) {
              DetId id1 = (*hit1d)->geographicalId();
              DTLayerId lid(id1.rawId());
              // Get the 1d DT RechHits from this layer
              DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);
              int layerHits = 0;
              for (DTRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
                double rhitDistance = fabs(ir->localPosition().x() - (**hit1d).localPosition().x());
                if (rhitDistance < coneSize)
                  layerHits++;
                LogTrace(theCategory) << "       " << (ir)->localPosition() << "  " << (**hit1d).localPosition()
                                      << " Distance: " << rhitDistance << " recHits: " << layerHits
                                      << "  SL: " << lid.superLayer() << endl;
              }
              if (layerHits > detRecHits)
                detRecHits = layerHits;
            }
          } else {
            DTLayerId lid(id.rawId());
            // Get the 1d DT RechHits from this layer
            DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);
            for (DTRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
              double rhitDistance = fabs(ir->localPosition().x() - (**imrh).localPosition().x());
              if (rhitDistance < coneSize)
                detRecHits++;
              LogTrace(theCategory) << "       " << (ir)->localPosition() << "  " << (**imrh).localPosition()
                                    << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
            }
          }
        }

      } else {
        DTLayerId lid(id.rawId());

        // Get the 1d DT RechHits from this layer
        DTRecHitCollection::range dRecHits = theDTRecHits->get(lid);

        for (DTRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
          double rhitDistance = fabs(ir->localPosition().x() - (**imrh).localPosition().x());
          if (rhitDistance < coneSize)
            detRecHits++;
          LogTrace(theCategory) << "       " << (ir)->localPosition() << "  " << (**imrh).localPosition()
                                << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
        }
      }
    }  // end of if DT
    else if (id.subdetId() == MuonSubdetId::CSC) {
      CSCDetId did(id.rawId());
      chamberId = did.chamberId();

      if ((*imrh)->recHits().size() > 1) {
        std::vector<const TrackingRecHit*> hits2d = (*imrh)->recHits();
        for (std::vector<const TrackingRecHit*>::const_iterator hit2d = hits2d.begin(); hit2d != hits2d.end();
             hit2d++) {
          DetId id1 = (*hit2d)->geographicalId();
          CSCDetId lid(id1.rawId());

          // Get the CSC Rechits from this layer
          CSCRecHit2DCollection::range dRecHits = theCSCRecHits->get(lid);
          int layerHits = 0;

          for (CSCRecHit2DCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
            double rhitDistance = (ir->localPosition() - (**hit2d).localPosition()).mag();
            if (rhitDistance < coneSize)
              layerHits++;
            LogTrace(theCategory) << ir->localPosition() << "  " << (**hit2d).localPosition()
                                  << " Distance: " << rhitDistance << " recHits: " << layerHits << endl;
          }
          if (layerHits > detRecHits)
            detRecHits = layerHits;
        }
      } else {
        // Get the CSC Rechits from this layer
        CSCRecHit2DCollection::range dRecHits = theCSCRecHits->get(did);

        for (CSCRecHit2DCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
          double rhitDistance = (ir->localPosition() - (**imrh).localPosition()).mag();
          if (rhitDistance < coneSize)
            detRecHits++;
          LogTrace(theCategory) << ir->localPosition() << "  " << (**imrh).localPosition()
                                << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
        }
      }
    }  //end of CSC if
    else if (id.subdetId() == MuonSubdetId::GEM) {
      GEMDetId did(id.rawId());
      chamberId = did.chamberId();

      if ((*imrh)->recHits().size() > 1) {
        std::vector<const TrackingRecHit*> hits2d = (*imrh)->recHits();
        for (std::vector<const TrackingRecHit*>::const_iterator hit2d = hits2d.begin(); hit2d != hits2d.end();
             hit2d++) {
          DetId id1 = (*hit2d)->geographicalId();
          GEMDetId lid(id1.rawId());

          // Get the GEM Rechits from this layer
          GEMRecHitCollection::range dRecHits = theGEMRecHits->get(lid);
          int layerHits = 0;

          for (GEMRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
            double rhitDistance = (ir->localPosition() - (**hit2d).localPosition()).mag();
            if (rhitDistance < coneSize)
              layerHits++;
            LogTrace(theCategory) << ir->localPosition() << "  " << (**hit2d).localPosition()
                                  << " Distance: " << rhitDistance << " recHits: " << layerHits << endl;
          }
          if (layerHits > detRecHits)
            detRecHits = layerHits;
        }
      } else {
        // Get the GEM Rechits from this layer
        GEMRecHitCollection::range dRecHits = theGEMRecHits->get(did);

        for (GEMRecHitCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
          double rhitDistance = (ir->localPosition() - (**imrh).localPosition()).mag();
          if (rhitDistance < coneSize)
            detRecHits++;
          LogTrace(theCategory) << ir->localPosition() << "  " << (**imrh).localPosition()
                                << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
        }
      }
    }  //end of GEM if
    else if (id.subdetId() == MuonSubdetId::ME0) {
      ME0DetId did(id.rawId());
      chamberId = did.chamberId();

      if ((*imrh)->recHits().size() > 1) {
        std::vector<const TrackingRecHit*> hits2d = (*imrh)->recHits();
        for (std::vector<const TrackingRecHit*>::const_iterator hit2d = hits2d.begin(); hit2d != hits2d.end();
             hit2d++) {
          DetId id1 = (*hit2d)->geographicalId();
          ME0DetId lid(id1.rawId());

          // Get the ME0 Rechits from this layer
          ME0SegmentCollection::range dRecHits = theME0RecHits->get(lid);
          int layerHits = 0;

          for (ME0SegmentCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
            double rhitDistance = (ir->localPosition() - (**hit2d).localPosition()).mag();
            if (rhitDistance < coneSize)
              layerHits++;
            LogTrace(theCategory) << ir->localPosition() << "  " << (**hit2d).localPosition()
                                  << " Distance: " << rhitDistance << " recHits: " << layerHits << endl;
          }
          if (layerHits > detRecHits)
            detRecHits = layerHits;
        }
      } else {
        // Get the ME0 Rechits from this layer
        ME0SegmentCollection::range dRecHits = theME0RecHits->get(did);

        for (ME0SegmentCollection::const_iterator ir = dRecHits.first; ir != dRecHits.second; ir++) {
          double rhitDistance = (ir->localPosition() - (**imrh).localPosition()).mag();
          if (rhitDistance < coneSize)
            detRecHits++;
          LogTrace(theCategory) << ir->localPosition() << "  " << (**imrh).localPosition()
                                << " Distance: " << rhitDistance << " recHits: " << detRecHits << endl;
        }
      }
    }  //end of ME0 if
    else {
      if (id.subdetId() != MuonSubdetId::RPC)
        LogError(theCategory) << " Wrong Hit Type ";
      continue;
    }

    map<DetId, int>::iterator imap = hitMap.find(chamberId);
    if (imap != hitMap.end()) {
      if (detRecHits > imap->second)
        imap->second = detRecHits;
    } else
      hitMap[chamberId] = detRecHits;

  }  // end of loop over muon rechits

  for (map<DetId, int>::iterator imap = hitMap.begin(); imap != hitMap.end(); imap++)
    LogTrace(theCategory) << " Station " << imap->first.rawId() << ": " << imap->second << endl;

  LogTrace(theCategory) << "CheckMuonHits: " << all.size();

  // check order of muon measurements
  if ((all.size() > 1) && (all.front()->globalPosition().mag() > all.back()->globalPosition().mag())) {
    LogTrace(theCategory) << "reverse order: ";
    stable_sort(all.begin(), all.end(), RecHitLessByDet(alongMomentum));
  }
}

//
// Get the hits from the first muon station (containing hits)
//
void GlobalMuonRefitter::getFirstHits(const reco::Track& muon,
                                      ConstRecHitContainer& all,
                                      ConstRecHitContainer& first) const {
  LogTrace(theCategory) << " GlobalMuonRefitter::getFirstHits\nall rechits length:" << all.size() << endl;
  first.clear();

  int station_to_keep = 999;
  vector<int> stations;
  for (ConstRecHitContainer::const_iterator ihit = all.begin(); ihit != all.end(); ++ihit) {
    int station = 0;
    bool use_it = true;
    DetId id = (*ihit)->geographicalId();
    unsigned raw_id = id.rawId();
    if (!(*ihit)->isValid())
      station = -1;
    else {
      if (id.det() == DetId::Muon) {
        switch (id.subdetId()) {
          case MuonSubdetId::DT:
            station = DTChamberId(raw_id).station();
            break;
          case MuonSubdetId::CSC:
            station = CSCDetId(raw_id).station();
            break;
          case MuonSubdetId::GEM:
            station = GEMDetId(raw_id).station();
            break;
          case MuonSubdetId::ME0:
            station = ME0DetId(raw_id).station();
            break;
          case MuonSubdetId::RPC:
            station = RPCDetId(raw_id).station();
            use_it = false;
            break;
        }
      }
    }

    if (use_it && station > 0 && station < station_to_keep)
      station_to_keep = station;
    stations.push_back(station);
    LogTrace(theCategory) << "rawId: " << raw_id << " station = " << station << " station_to_keep is now "
                          << station_to_keep;
  }

  if (station_to_keep <= 0 || station_to_keep > 4 || stations.size() != all.size())
    LogInfo(theCategory) << "failed to getFirstHits (all muon hits are outliers/bad ?)! station_to_keep = "
                         << station_to_keep << " stations.size " << stations.size() << " all.size " << all.size();

  for (unsigned i = 0; i < stations.size(); ++i)
    if (stations[i] >= 0 && stations[i] <= station_to_keep)
      first.push_back(all[i]);

  return;
}

//
// select muon hits compatible with trajectory;
// check hits in chambers with showers
//
GlobalMuonRefitter::ConstRecHitContainer GlobalMuonRefitter::selectMuonHits(const Trajectory& traj,
                                                                            const map<DetId, int>& hitMap) const {
  ConstRecHitContainer muonRecHits;
  const double globalChi2Cut = 200.0;

  vector<TrajectoryMeasurement> muonMeasurements = traj.measurements();

  // loop through all muon hits and skip hits with bad chi2 in chambers with high occupancy
  for (std::vector<TrajectoryMeasurement>::const_iterator im = muonMeasurements.begin(); im != muonMeasurements.end();
       im++) {
    if (!(*im).recHit()->isValid())
      continue;
    if ((*im).recHit()->det()->geographicalId().det() != DetId::Muon) {
      //      if ( ( chi2ndf < globalChi2Cut ) )
      muonRecHits.push_back((*im).recHit());
      continue;
    }
    const MuonTransientTrackingRecHit* immrh = dynamic_cast<const MuonTransientTrackingRecHit*>((*im).recHit().get());

    DetId id = immrh->geographicalId();
    DetId chamberId;
    int threshold = 0;
    double chi2Cut = 0.0;

    // get station of hit if it is in DT
    if ((*immrh).isDT()) {
      DTChamberId did(id.rawId());
      chamberId = did;
      threshold = theHitThreshold;
      chi2Cut = theDTChi2Cut;
    }
    // get station of hit if it is in CSC
    else if ((*immrh).isCSC()) {
      CSCDetId did(id.rawId());
      chamberId = did.chamberId();
      threshold = theHitThreshold;
      chi2Cut = theCSCChi2Cut;
    }
    // get station of hit if it is in GEM
    else if ((*immrh).isGEM()) {
      GEMDetId did(id.rawId());
      chamberId = did.chamberId();
      threshold = theHitThreshold;
      chi2Cut = theGEMChi2Cut;
    }
    // get station of hit if it is in ME0
    else if ((*immrh).isME0()) {
      ME0DetId did(id.rawId());
      chamberId = did.chamberId();
      threshold = theHitThreshold;
      chi2Cut = theME0Chi2Cut;
    }
    // get station of hit if it is in RPC
    else if ((*immrh).isRPC()) {
      RPCDetId rpcid(id.rawId());
      chamberId = rpcid;
      threshold = theHitThreshold;
      chi2Cut = theRPCChi2Cut;
    } else
      continue;

    double chi2ndf = (*im).estimate() / (*im).recHit()->dimension();

    bool keep = true;
    map<DetId, int>::const_iterator imap = hitMap.find(chamberId);
    if (imap != hitMap.end())
      if (imap->second > threshold)
        keep = false;

    if ((keep || (chi2ndf < chi2Cut)) && (chi2ndf < globalChi2Cut)) {
      muonRecHits.push_back((*im).recHit());
    } else {
      LogTrace(theCategory) << "Skip hit: " << id.rawId() << " chi2=" << chi2ndf << " ( threshold: " << chi2Cut
                            << ") Det: " << imap->second << endl;
    }
  }

  // check order of rechits
  reverse(muonRecHits.begin(), muonRecHits.end());
  return muonRecHits;
}

//
// print RecHits
//
void GlobalMuonRefitter::printHits(const ConstRecHitContainer& hits) const {
  LogTrace(theCategory) << "Used RecHits: " << hits.size();
  for (ConstRecHitContainer::const_iterator ir = hits.begin(); ir != hits.end(); ir++) {
    if (!(*ir)->isValid()) {
      LogTrace(theCategory) << "invalid RecHit";
      continue;
    }

    const GlobalPoint& pos = (*ir)->globalPosition();

    LogTrace(theCategory) << "r = " << sqrt(pos.x() * pos.x() + pos.y() * pos.y()) << "  z = " << pos.z()
                          << "  dimension = " << (*ir)->dimension()
                          << "  det = " << (*ir)->det()->geographicalId().det()
                          << "  subdet = " << (*ir)->det()->subDetector()
                          << "  raw id = " << (*ir)->det()->geographicalId().rawId();
  }
}

//
// add Trajectory* to TrackCand if not already present
//
GlobalMuonRefitter::RefitDirection GlobalMuonRefitter::checkRecHitsOrdering(
    const TransientTrackingRecHit::ConstRecHitContainer& recHits) const {
  if (!recHits.empty()) {
    ConstRecHitContainer::const_iterator frontHit = recHits.begin();
    ConstRecHitContainer::const_iterator backHit = recHits.end() - 1;
    while (!(*frontHit)->isValid() && frontHit != backHit) {
      frontHit++;
    }
    while (!(*backHit)->isValid() && backHit != frontHit) {
      backHit--;
    }

    double rFirst = (*frontHit)->globalPosition().mag();
    double rLast = (*backHit)->globalPosition().mag();

    if (rFirst < rLast)
      return insideOut;
    else if (rFirst > rLast)
      return outsideIn;
    else {
      LogError(theCategory) << "Impossible determine the rechits order" << endl;
      return undetermined;
    }
  } else {
    LogError(theCategory) << "Impossible determine the rechits order" << endl;
    return undetermined;
  }
}

//
// Convert Tracks into Trajectories with a given set of hits
//
vector<Trajectory> GlobalMuonRefitter::transform(
    const reco::Track& newTrack,
    const reco::TransientTrack track,
    const TransientTrackingRecHit::ConstRecHitContainer& urecHitsForReFit) const {
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = urecHitsForReFit;
  LogTrace(theCategory) << "GlobalMuonRefitter::transform: " << recHitsForReFit.size() << " hits:";
  printHits(recHitsForReFit);

  if (recHitsForReFit.size() < 2)
    return vector<Trajectory>();

  // Check the order of the rechits
  RefitDirection recHitsOrder = checkRecHitsOrdering(recHitsForReFit);

  LogTrace(theCategory) << "checkRecHitsOrdering() returned " << recHitsOrder << ", theRefitDirection is "
                        << theRefitDirection << " (insideOut == " << insideOut << ", outsideIn == " << outsideIn << ")";

  // Reverse the order in the case of inconsistency between the fit direction and the rechit order
  if (theRefitDirection != recHitsOrder)
    reverse(recHitsForReFit.begin(), recHitsForReFit.end());

  // Even though we checked the rechits' ordering above, we may have
  // already flipped them elsewhere (getFirstHits() is such a
  // culprit). Use the global positions of the states and the desired
  // refit direction to find the starting TSOS.
  TrajectoryStateOnSurface firstTSOS, lastTSOS;
  unsigned int innerId;  //UNUSED: outerId;
  bool order_swapped = track.outermostMeasurementState().globalPosition().mag() <
                       track.innermostMeasurementState().globalPosition().mag();
  bool inner_is_first;
  LogTrace(theCategory) << "order swapped? " << order_swapped;

  // Fill the starting state, depending on the ordering above.
  if ((theRefitDirection == insideOut && !order_swapped) || (theRefitDirection == outsideIn && order_swapped)) {
    innerId = newTrack.innerDetId();
    //UNUSED:    outerId   = newTrack.outerDetId();
    firstTSOS = track.innermostMeasurementState();
    lastTSOS = track.outermostMeasurementState();
    inner_is_first = true;
  } else {
    innerId = newTrack.outerDetId();
    //UNUSED:    outerId   = newTrack.innerDetId();
    firstTSOS = track.outermostMeasurementState();
    lastTSOS = track.innermostMeasurementState();
    inner_is_first = false;
  }

  LogTrace(theCategory) << "firstTSOS: inner_is_first? " << inner_is_first << " globalPosition is "
                        << firstTSOS.globalPosition() << " innerId is " << innerId;

  if (!firstTSOS.isValid()) {
    LogWarning(theCategory) << "Error wrong initial state!" << endl;
    return vector<Trajectory>();
  }

  firstTSOS.rescaleError(theRescaleErrorFactor);

  // This is the only way to get a TrajectorySeed with settable propagation direction
  PTrajectoryStateOnDet garbage1;
  edm::OwnVector<TrackingRecHit> garbage2;

  // These lines cause the code to ignore completely what was set
  // above, and force propDir for tracks from collisions!
  //  if(propDir == alongMomentum && theRefitDirection == outsideIn)  propDir=oppositeToMomentum;
  //  if(propDir == oppositeToMomentum && theRefitDirection == insideOut) propDir=alongMomentum;

  const TrajectoryStateOnSurface& tsosForDir = inner_is_first ? lastTSOS : firstTSOS;
  PropagationDirection propDir =
      (tsosForDir.globalPosition().basicVector().dot(tsosForDir.globalMomentum().basicVector()) > 0)
          ? alongMomentum
          : oppositeToMomentum;
  LogTrace(theCategory) << "propDir based on firstTSOS x dot p is " << propDir << " (alongMomentum == " << alongMomentum
                        << ", oppositeToMomentum == " << oppositeToMomentum << ")";

  // Additional propagation diretcion determination logic for cosmic muons
  if (theCosmicFlag) {
    PropagationDirection propDir_first =
        (firstTSOS.globalPosition().basicVector().dot(firstTSOS.globalMomentum().basicVector()) > 0)
            ? alongMomentum
            : oppositeToMomentum;
    PropagationDirection propDir_last =
        (lastTSOS.globalPosition().basicVector().dot(lastTSOS.globalMomentum().basicVector()) > 0) ? alongMomentum
                                                                                                   : oppositeToMomentum;
    LogTrace(theCategory) << "propDir_first " << propDir_first << ", propdir_last " << propDir_last << " : they "
                          << (propDir_first == propDir_last ? "agree" : "disagree");

    int y_count = 0;
    for (TransientTrackingRecHit::ConstRecHitContainer::const_iterator it = recHitsForReFit.begin();
         it != recHitsForReFit.end();
         ++it) {
      if ((*it)->globalPosition().y() > 0)
        ++y_count;
      else
        --y_count;
    }

    PropagationDirection propDir_ycount = alongMomentum;
    if (y_count > 0) {
      if (theRefitDirection == insideOut)
        propDir_ycount = oppositeToMomentum;
      else if (theRefitDirection == outsideIn)
        propDir_ycount = alongMomentum;
    } else {
      if (theRefitDirection == insideOut)
        propDir_ycount = alongMomentum;
      else if (theRefitDirection == outsideIn)
        propDir_ycount = oppositeToMomentum;
    }

    LogTrace(theCategory) << "y_count = " << y_count << "; based on geometrically-outermost TSOS, propDir is "
                          << propDir << ": " << (propDir == propDir_ycount ? "agrees" : "disagrees")
                          << " with ycount determination";

    if (propDir_first != propDir_last) {
      LogTrace(theCategory) << "since first/last disagreed, using y_count propDir";
      propDir = propDir_ycount;
    }
  }

  TrajectorySeed seed(garbage1, garbage2, propDir);

  if (recHitsForReFit.front()->geographicalId() != DetId(innerId)) {
    LogDebug(theCategory) << "Propagation occured" << endl;
    LogTrace(theCategory) << "propagating firstTSOS at " << firstTSOS.globalPosition()
                          << " to first rechit with surface pos "
                          << recHitsForReFit.front()->det()->surface().toGlobal(LocalPoint(0, 0, 0));
    firstTSOS =
        theService->propagator(thePropagatorName)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    if (!firstTSOS.isValid()) {
      LogDebug(theCategory) << "Propagation error!" << endl;
      return vector<Trajectory>();
    }
  }

  LogDebug(theCategory) << " GlobalMuonRefitter : theFitter " << propDir << endl;
  LogDebug(theCategory) << "                      First TSOS: " << firstTSOS.globalPosition()
                        << "  p=" << firstTSOS.globalMomentum() << " = " << firstTSOS.globalMomentum().mag() << endl;

  LogDebug(theCategory) << "                      Starting seed: "
                        << " nHits= " << seed.nHits() << " tsos: " << seed.startingState().parameters().position()
                        << "  p=" << seed.startingState().parameters().momentum() << endl;

  LogDebug(theCategory) << "                      RecHits: " << recHitsForReFit.size() << endl;

  vector<Trajectory> trajectories = theFitter->fit(seed, recHitsForReFit, firstTSOS);

  if (trajectories.empty()) {
    LogDebug(theCategory) << "No Track refitted!" << endl;
    return vector<Trajectory>();
  }
  return trajectories;
}

//
// Remove Selected Station Rec Hits
//
GlobalMuonRefitter::ConstRecHitContainer GlobalMuonRefitter::getRidOfSelectStationHits(
    const ConstRecHitContainer& hits, const TrackerTopology* tTopo) const {
  ConstRecHitContainer results;
  ConstRecHitContainer::const_iterator it = hits.begin();
  for (; it != hits.end(); it++) {
    DetId id = (*it)->geographicalId();

    //Check that this is a Muon hit that we're toying with -- else pass on this because the hacker is a moron / not careful

    if (id.det() == DetId::Tracker && theTrackerSkipSystem > 0) {
      int layer = -999;
      int disk = -999;
      int wheel = -999;
      if (id.subdetId() == theTrackerSkipSystem) {
        //                              continue;  //caveat that just removes the whole system from refitting

        if (theTrackerSkipSystem == PXB) {
          layer = tTopo->pxbLayer(id);
        }
        if (theTrackerSkipSystem == TIB) {
          layer = tTopo->tibLayer(id);
        }

        if (theTrackerSkipSystem == TOB) {
          layer = tTopo->tobLayer(id);
        }
        if (theTrackerSkipSystem == PXF) {
          disk = tTopo->pxfDisk(id);
        }
        if (theTrackerSkipSystem == TID) {
          wheel = tTopo->tidWheel(id);
        }
        if (theTrackerSkipSystem == TEC) {
          wheel = tTopo->tecWheel(id);
        }
        if (theTrackerSkipSection >= 0 && layer == theTrackerSkipSection)
          continue;
        if (theTrackerSkipSection >= 0 && disk == theTrackerSkipSection)
          continue;
        if (theTrackerSkipSection >= 0 && wheel == theTrackerSkipSection)
          continue;
      }
    }

    if (id.det() == DetId::Muon && theSkipStation) {
      int station = -999;
      //UNUSED:      int wheel = -999;
      if (id.subdetId() == MuonSubdetId::DT) {
        DTChamberId did(id.rawId());
        station = did.station();
        //UNUSED:	wheel = did.wheel();
      } else if (id.subdetId() == MuonSubdetId::CSC) {
        CSCDetId did(id.rawId());
        station = did.station();
      } else if (id.subdetId() == MuonSubdetId::GEM) {
        GEMDetId did(id.rawId());
        station = did.station();
      } else if (id.subdetId() == MuonSubdetId::ME0) {
        ME0DetId did(id.rawId());
        station = did.station();
      } else if (id.subdetId() == MuonSubdetId::RPC) {
        RPCDetId rpcid(id.rawId());
        station = rpcid.station();
      }
      if (station == theSkipStation)
        continue;
    }
    results.push_back(*it);
  }
  return results;
}
