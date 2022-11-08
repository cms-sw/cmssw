// -*- C++ -*-
//
// Package:    CSCOverlapsTrackPreparation
// Class:      CSCOverlapsTrackPreparation
//
/**\class CSCOverlapsTrackPreparation CSCOverlapsTrackPreparation.cc Alignment/CSCOverlapsTrackPreparation/src/CSCOverlapsTrackPreparation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: CSCOverlapsTrackPreparation.cc,v 1.8 2011/03/22 09:49:50 innocent Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class CSCOverlapsTrackPreparation : public edm::one::EDProducer<> {
public:
  explicit CSCOverlapsTrackPreparation(const edm::ParameterSet&);
  ~CSCOverlapsTrackPreparation() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  enum { kNothing, kSimpleFit, kAllButOne, kExtrapolate };

  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  const edm::InputTag m_src;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> globalGeomToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSCOverlapsTrackPreparation::CSCOverlapsTrackPreparation(const edm::ParameterSet& iConfig)
    : m_src(iConfig.getParameter<edm::InputTag>("src")),
      cscGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()),
      globalGeomToken_(esConsumes<edm::Transition::BeginRun>()),
      trackToken_(consumes<reco::TrackCollection>(m_src)) {
  produces<std::vector<Trajectory>>();
  produces<TrajTrackAssociationCollection>();
}

//
// member functions
//
void CSCOverlapsTrackPreparation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("ALCARECOMuAlBeamHaloOverlaps"));
  descriptions.add("cscOverlapsTrackPreparation", desc);
}

// ------------ method called to produce the data  ------------
void CSCOverlapsTrackPreparation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const edm::Handle<reco::TrackCollection>& tracks = iEvent.getHandle(trackToken_);

  const CSCGeometry* cscGeometry = &iSetup.getData(cscGeomToken_);
  const MagneticField* magneticField = &iSetup.getData(magneticFieldToken_);
  const GlobalTrackingGeometry* globalGeometry = &iSetup.getData(globalGeomToken_);

  MuonTransientTrackingRecHitBuilder muonTransBuilder;

  // Create a collection of Trajectories, to put in the Event
  auto trajectoryCollection = std::make_unique<std::vector<Trajectory>>();

  // Remember which trajectory is associated with which track
  std::map<edm::Ref<std::vector<Trajectory>>::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
  edm::Ref<std::vector<Trajectory>>::key_type trajCounter = 0;
  edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;

  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
    trackCounter++;

    // now we'll actually put hits on the new trajectory
    // these must be in lock-step
    edm::OwnVector<TrackingRecHit> clonedHits;
    std::vector<TrajectoryMeasurement::ConstRecHitPointer> transHits;
    std::vector<TrajectoryStateOnSurface> TSOSes;

    for (auto const& hit : track->recHits()) {
      DetId id = hit->geographicalId();
      if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC) {
        const Surface& layerSurface = cscGeometry->idToDet(id)->surface();
        TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(hit, globalGeometry));

        AlgebraicVector5 params;  // meaningless, CSCOverlapsAlignmentAlgorithm does the fit internally
        params[0] = 1.;           // straight-forward direction
        params[1] = 0.;
        params[2] = 0.;
        params[3] = 0.;  // center of the chamber
        params[4] = 0.;
        LocalTrajectoryParameters localTrajectoryParameters(params, 1., false);
        LocalTrajectoryError localTrajectoryError(0.001, 0.001, 0.001, 0.001, 0.001);

        // these must be in lock-step
        clonedHits.push_back(hit->clone());
        transHits.push_back(hitPtr);
        TSOSes.push_back(
            TrajectoryStateOnSurface(localTrajectoryParameters, localTrajectoryError, layerSurface, &*magneticField));
      }  // end if CSC
    }    // end loop over hits

    assert(clonedHits.size() == transHits.size());
    assert(transHits.size() == TSOSes.size());

    // build the trajectory
    if (!clonedHits.empty()) {
      PTrajectoryStateOnDet const PTraj =
          trajectoryStateTransform::persistentState(*(TSOSes.begin()), clonedHits.begin()->geographicalId().rawId());
      TrajectorySeed trajectorySeed(PTraj, clonedHits, alongMomentum);
      Trajectory trajectory(trajectorySeed, alongMomentum);

      edm::OwnVector<TrackingRecHit>::const_iterator clonedHit = clonedHits.begin();
      std::vector<TrajectoryMeasurement::ConstRecHitPointer>::const_iterator transHitPtr = transHits.begin();
      std::vector<TrajectoryStateOnSurface>::const_iterator TSOS = TSOSes.begin();
      for (; clonedHit != clonedHits.end(); ++clonedHit, ++transHitPtr, ++TSOS) {
        trajectory.push(TrajectoryMeasurement(*TSOS, *TSOS, *TSOS, (*transHitPtr)));
      }

      trajectoryCollection->push_back(trajectory);

      // Remember which Trajectory is associated with which Track
      trajCounter++;
      reference_map[trajCounter] = trackCounter;

    }  // end if there are any clonedHits/TSOSes to work with
  }    // end loop over tracks

  unsigned int numTrajectories = trajectoryCollection->size();

  // insert the trajectories into the Event
  edm::OrphanHandle<std::vector<Trajectory>> ohTrajs = iEvent.put(std::move(trajectoryCollection));

  // create the trajectory <-> track association map
  auto trajTrackMap = std::make_unique<TrajTrackAssociationCollection>();

  for (trajCounter = 0; trajCounter < numTrajectories; trajCounter++) {
    edm::Ref<reco::TrackCollection>::key_type trackCounter = reference_map[trajCounter];

    trajTrackMap->insert(edm::Ref<std::vector<Trajectory>>(ohTrajs, trajCounter),
                         edm::Ref<reco::TrackCollection>(tracks, trackCounter));
  }
  // and put it in the Event, also
  iEvent.put(std::move(trajTrackMap));
}

// ------------ method called once each job just before starting event loop  ------------
void CSCOverlapsTrackPreparation::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CSCOverlapsTrackPreparation::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCOverlapsTrackPreparation);
