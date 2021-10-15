// -*- C++ -*-
//
// Package:    StandAloneTest
// Class:      StandAloneTest
//
/**\class StandAloneTest StandAloneTest.cc Dummy/StandAloneTest/src/StandAloneTest.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Sep 26 02:50:24 CEST 2009
// $Id: StandAloneTest.cc,v 1.2 2010/01/06 15:38:44 mussgill Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFromTrack.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

//
// class decleration
//

class StandAloneTest : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit StandAloneTest(const edm::ParameterSet &);
  ~StandAloneTest();

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event &, const edm::EventSetup &);
  virtual void endJob();

  // ----------member data ---------------------------

  const edm::InputTag m_Tracks;
  const edm::EDGetTokenT<reco::TrackCollection> m_tracksToken;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> m_trajtracksmapToken;

  const MuonResidualsFromTrack::BuilderToken m_builderToken;
  const edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> m_globalGeometryToken;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> m_cscGeometryToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> m_propToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magneticFieldToken;
  const edm::ESGetToken<DetIdAssociator, DetIdAssociatorRecord> m_muonDetIdAssociatorToken;

  // declare the TTree
  TTree *m_ttree;
  Int_t m_ttree_station;
  Int_t m_ttree_chamber;
  Float_t m_ttree_resid;
  Float_t m_ttree_residslope;
  Float_t m_ttree_phi;
  Float_t m_ttree_qoverpt;

  MuonAlignment *m_muonAlignment;
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
StandAloneTest::StandAloneTest(const edm::ParameterSet &iConfig)
    : m_Tracks(iConfig.getParameter<edm::InputTag>("Tracks")),
      m_tracksToken(consumes(m_Tracks)),
      m_trajtracksmapToken(consumes(edm::InputTag("TrackRefitter", "Refitted"))),
      m_builderToken(esConsumes(MuonResidualsFromTrack::builderESInputTag())),
      m_globalGeometryToken(esConsumes()),
      m_cscGeometryToken(esConsumes()),
      m_propToken(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))),
      m_magneticFieldToken(esConsumes()),
      m_muonDetIdAssociatorToken(esConsumes(edm::ESInputTag("", "MuonDetIdAssociator"))) {
  edm::Service<TFileService> tFileService;
  usesResource(TFileService::kSharedResource);

  // book the TTree
  m_ttree = tFileService->make<TTree>("ttree", "ttree");
  m_ttree->Branch("station", &m_ttree_station, "station/I");
  m_ttree->Branch("chamber", &m_ttree_chamber, "chamber/I");
  m_ttree->Branch("resid", &m_ttree_resid, "resid/F");
  m_ttree->Branch("residslope", &m_ttree_residslope, "residslope/F");
  m_ttree->Branch("phi", &m_ttree_phi, "phi/F");
  m_ttree->Branch("qoverpt", &m_ttree_qoverpt, "qoverpt/F");

  m_muonAlignment = NULL;
}

StandAloneTest::~StandAloneTest() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void StandAloneTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // create a muon alignment object ONCE (not used for much, only a formalilty for MuonResidualsFromTrack)
  if (m_muonAlignment == NULL) {
    m_muonAlignment = new MuonAlignment(iSetup);
  }

  // get tracks and refitted from the Event
  edm::Handle<reco::TrackCollection> tracks = iEvent.getHandle(m_tracksToken);
  edm::Handle<TrajTrackAssociationCollection> trajtracksmap = iEvent.getHandle(m_trajtracksmapToken);

  // get all tracking and CSC geometries from the EventSetup
  edm::ESHandle<GlobalTrackingGeometry> globalGeometry = iSetup.getHandle(m_globalGeometryToken);
  edm::ESHandle<CSCGeometry> cscGeometry = iSetup.getHandle(m_cscGeometryToken);

  auto builder = iSetup.getHandle(m_builderToken);
  edm::ESHandle<Propagator> prop = iSetup.getHandle(m_propToken);
  edm::ESHandle<MagneticField> magneticField = iSetup.getHandle(m_magneticFieldToken);
  edm::ESHandle<DetIdAssociator> muonDetIdAssociator_ = iSetup.getHandle(m_muonDetIdAssociatorToken);

  // loop over tracks
  for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
    // find the corresponding refitted trajectory
    const Trajectory *traj = NULL;
    for (TrajTrackAssociationCollection::const_iterator iPair = trajtracksmap->begin(); iPair != trajtracksmap->end();
         ++iPair) {
      if (&(*(iPair->val)) == &(*track)) {
        traj = &(*(iPair->key));
      }
    }

    // if good track, good trajectory
    if (track->pt() > 20. && traj != NULL && traj->isValid()) {
      // calculate all residuals on this track
      MuonResidualsFromTrack muonResidualsFromTrack(builder,
                                                    magneticField,
                                                    globalGeometry,
                                                    muonDetIdAssociator_,
                                                    prop,
                                                    traj,
                                                    &(*track),
                                                    m_muonAlignment->getAlignableNavigator(),
                                                    1000.);
      std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

      // if the tracker part of refit is okay
      if (muonResidualsFromTrack.trackerNumHits() >= 10 && muonResidualsFromTrack.trackerRedChi2() < 10.) {
        // loop over ALL chambers
        for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin(); chamberId != chamberIds.end();
             ++chamberId) {
          // if CSC
          if (chamberId->det() == DetId::Muon && chamberId->subdetId() == MuonSubdetId::CSC) {
            CSCDetId cscid(chamberId->rawId());
            int station = (cscid.endcap() == 1 ? 1 : -1) * (10 * cscid.station() + cscid.ring());
            MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);

            // if this segment is okay and has 6 hits
            if (csc != NULL && csc->numHits() >= 6) {
              // fill the TTree
              m_ttree_station = station;
              m_ttree_chamber = cscid.chamber();
              m_ttree_resid = csc->residual();
              m_ttree_residslope = csc->resslope();
              m_ttree_phi = csc->global_trackpos().phi();
              m_ttree_qoverpt = double(track->charge()) / track->pt();
              m_ttree->Fill();
            }  // end if CSC is okay

          }  // end if CSC

        }  // end loop over all chambers

      }  // end if tracker part of refit is okay

    }  // end if good track, good track refit

  }  // end loop over tracks
}

// ------------ method called once each job just before starting event loop  ------------
void StandAloneTest::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void StandAloneTest::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(StandAloneTest);
