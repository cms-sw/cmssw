// -*- C++ -*-
//
// Package:    EgammaHLTAlgos
// Class:      EgammaHLTPixelMatchElectronAlgo.
//
/**\class EgammaHLTPixelMatchElectronAlgo EgammaHLTAlgos/EgammaHLTPixelMatchElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking for HLT
*/
//
// Original Author:  Monica Vazquez Acosta (CERN)
//
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"

#include "EgammaHLTPixelMatchElectronAlgo.h"

using namespace edm;
using namespace std;
using namespace reco;

EgammaHLTPixelMatchElectronAlgo::EgammaHLTPixelMatchElectronAlgo(const edm::ParameterSet& conf,
                                                                 edm::ConsumesCollector&& iC)
    : trackProducer_(iC.consumes(conf.getParameter<edm::InputTag>("TrackProducer"))),
      gsfTrackProducer_(iC.consumes(conf.getParameter<edm::InputTag>("GsfTrackProducer"))),
      useGsfTracks_(conf.getParameter<bool>("UseGsfTracks")),
      bsProducer_(iC.consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BSProducer"))),
      magneticFieldToken_(iC.esConsumes()),
      trackerGeometryToken_(iC.esConsumes()) {}

void EgammaHLTPixelMatchElectronAlgo::setupES(const edm::EventSetup& eventSetup) {
  //services
  bool updateField = magneticFieldWatcher_.check(eventSetup);
  bool updateGeometry = trackerGeometryWatcher_.check(eventSetup);

  if (updateField) {
    magField_ = eventSetup.getHandle(magneticFieldToken_);
  }

  if (useGsfTracks_) {  //only need the geom and mtsTransform if we are doing gsf tracks
    if (updateGeometry) {
      trackerGeom_ = eventSetup.getHandle(trackerGeometryToken_);
    }
    if (updateField || updateGeometry || !mtsTransform_) {
      mtsTransform_ = std::make_unique<MultiTrajectoryStateTransform>(trackerGeom_.product(), magField_.product());
    }
  }
}

void EgammaHLTPixelMatchElectronAlgo::run(Event& e, ElectronCollection& outEle) {
  // get the input
  edm::Handle<TrackCollection> tracksH;
  if (!useGsfTracks_)
    e.getByToken(trackProducer_, tracksH);

  // get the input
  edm::Handle<GsfTrackCollection> gsfTracksH;
  if (useGsfTracks_)
    e.getByToken(gsfTrackProducer_, gsfTracksH);

  //Get the Beam Spot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  e.getByToken(bsProducer_, recoBeamSpotHandle);

  // gets its position
  const BeamSpot::Point& bsPosition = recoBeamSpotHandle->position();
  Global3DPoint bs(bsPosition.x(), bsPosition.y(), 0);
  process(tracksH, gsfTracksH, outEle, bs);

  return;
}

void EgammaHLTPixelMatchElectronAlgo::process(edm::Handle<TrackCollection> tracksH,
                                              edm::Handle<GsfTrackCollection> gsfTracksH,
                                              ElectronCollection& outEle,
                                              Global3DPoint& bs) {
  if (!useGsfTracks_) {
    for (unsigned int i = 0; i < tracksH->size(); ++i) {
      const TrackRef trackRef = edm::Ref<TrackCollection>(tracksH, i);
      edm::RefToBase<TrajectorySeed> seed = trackRef->extra()->seedRef();
      ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();

      edm::RefToBase<CaloCluster> caloCluster = elseed->caloCluster();
      SuperClusterRef scRef = caloCluster.castTo<SuperClusterRef>();

      // Get the momentum at vertex (not at the innermost layer)
      TSCPBuilderNoMaterial tscpBuilder;

      FreeTrajectoryState fts = trajectoryStateTransform::innerFreeState(*trackRef, magField_.product());
      TrajectoryStateClosestToPoint tscp = tscpBuilder(fts, bs);

      float scale = scRef->energy() / tscp.momentum().mag();

      const math::XYZTLorentzVector momentum(
          tscp.momentum().x() * scale, tscp.momentum().y() * scale, tscp.momentum().z() * scale, scRef->energy());

      Electron ele(trackRef->charge(), momentum, trackRef->vertex());
      ele.setSuperCluster(scRef);
      edm::Ref<TrackCollection> myRef(tracksH, i);
      ele.setTrack(myRef);
      outEle.push_back(ele);
    }  // loop over tracks
  } else {
    // clean gsf tracks
    std::vector<unsigned int> flag(gsfTracksH->size(), 0);
    if (gsfTracksH->empty())
      return;

    for (unsigned int i = 0; i < gsfTracksH->size() - 1; ++i) {
      const GsfTrackRef trackRef1 = edm::Ref<GsfTrackCollection>(gsfTracksH, i);
      ElectronSeedRef elseed1 = trackRef1->extra()->seedRef().castTo<ElectronSeedRef>();
      SuperClusterRef scRef1 = elseed1->caloCluster().castTo<SuperClusterRef>();

      TrajectoryStateOnSurface inTSOS = mtsTransform_->innerStateOnSurface((*trackRef1));
      TrajectoryStateOnSurface fts = mtsTransform_->extrapolatedState(inTSOS, bs);
      GlobalVector innMom;
      float pin1 = trackRef1->pMode();
      if (fts.isValid()) {
        multiTrajectoryStateMode::momentumFromModeCartesian(fts, innMom);
        pin1 = innMom.mag();
      }

      for (unsigned int j = i + 1; j < gsfTracksH->size(); ++j) {
        const GsfTrackRef trackRef2 = edm::Ref<GsfTrackCollection>(gsfTracksH, j);
        ElectronSeedRef elseed2 = trackRef2->extra()->seedRef().castTo<ElectronSeedRef>();
        SuperClusterRef scRef2 = elseed2->caloCluster().castTo<SuperClusterRef>();

        TrajectoryStateOnSurface inTSOS = mtsTransform_->innerStateOnSurface((*trackRef2));
        TrajectoryStateOnSurface fts = mtsTransform_->extrapolatedState(inTSOS, bs);
        GlobalVector innMom;
        float pin2 = trackRef2->pMode();
        if (fts.isValid()) {
          multiTrajectoryStateMode::momentumFromModeCartesian(fts, innMom);
          pin2 = innMom.mag();
        }

        if (scRef1 == scRef2) {
          bool isSameLayer = false;
          bool iGsfInnermostWithLostHits = isInnerMostWithLostHits(trackRef2, trackRef1, isSameLayer);

          if (iGsfInnermostWithLostHits) {
            flag[j] = 1;
          } else if (isSameLayer) {
            if (fabs((scRef1->energy() / pin1) - 1) < fabs((scRef2->energy() / pin2) - 1))
              flag[j] = 1;
          } else {
            flag[i] = 1;
          }
        }
      }
    }

    for (unsigned int i = 0; i < gsfTracksH->size(); ++i) {
      if (flag[i] == 1)
        continue;

      const GsfTrackRef trackRef = edm::Ref<GsfTrackCollection>(gsfTracksH, i);
      ElectronSeedRef elseed = trackRef->extra()->seedRef().castTo<ElectronSeedRef>();
      SuperClusterRef scRef = elseed->caloCluster().castTo<SuperClusterRef>();

      // Get the momentum at vertex (not at the innermost layer)
      TrajectoryStateOnSurface inTSOS = mtsTransform_->innerStateOnSurface((*trackRef));
      TrajectoryStateOnSurface fts = mtsTransform_->extrapolatedState(inTSOS, bs);
      GlobalVector innMom;
      multiTrajectoryStateMode::momentumFromModeCartesian(inTSOS, innMom);
      if (fts.isValid()) {
        multiTrajectoryStateMode::momentumFromModeCartesian(fts, innMom);
      }

      float scale = scRef->energy() / innMom.mag();
      const math::XYZTLorentzVector momentum(
          innMom.x() * scale, innMom.y() * scale, innMom.z() * scale, scRef->energy());

      Electron ele(trackRef->charge(), momentum, trackRef->vertex());
      ele.setSuperCluster(scRef);
      edm::Ref<GsfTrackCollection> myRef(gsfTracksH, i);
      ele.setGsfTrack(myRef);
      outEle.push_back(ele);
    }
  }
}

bool EgammaHLTPixelMatchElectronAlgo::isInnerMostWithLostHits(const reco::GsfTrackRef& nGsfTrack,
                                                              const reco::GsfTrackRef& iGsfTrack,
                                                              bool& sameLayer) {
  // define closest using the lost hits on the expectedhitsineer
  auto nLostHits = nGsfTrack->missingInnerHits();
  auto iLostHits = iGsfTrack->missingInnerHits();

  if (nLostHits != iLostHits) {
    return (nLostHits > iLostHits);
  } else {
    sameLayer = true;
    return false;
  }
}
