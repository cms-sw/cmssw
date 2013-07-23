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
// $Id: StandAloneTest.cc,v 1.3 2011/10/12 23:05:21 khotilov Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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

//
// class decleration
//

class StandAloneTest : public edm::EDAnalyzer {
   public:
      explicit StandAloneTest(const edm::ParameterSet&);
      ~StandAloneTest();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

      edm::InputTag m_Tracks;

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
StandAloneTest::StandAloneTest(const edm::ParameterSet& iConfig)
   : m_Tracks(iConfig.getParameter<edm::InputTag>("Tracks"))
{
   edm::Service<TFileService> tFileService;

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


StandAloneTest::~StandAloneTest()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
StandAloneTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // create a muon alignment object ONCE (not used for much, only a formalilty for MuonResidualsFromTrack)
   if (m_muonAlignment == NULL) {
      m_muonAlignment = new MuonAlignment(iSetup);
   }

   // get tracks and refitted from the Event
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_Tracks, tracks);
   edm::Handle<TrajTrackAssociationCollection> trajtracksmap;
   iEvent.getByLabel("TrackRefitter", "Refitted", trajtracksmap);

   // get all tracking and CSC geometries from the EventSetup
   edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);
   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   // loop over tracks
   for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      // find the corresponding refitted trajectory
      const Trajectory *traj = NULL;
      for (TrajTrackAssociationCollection::const_iterator iPair = trajtracksmap->begin();  iPair != trajtracksmap->end();  ++iPair) {
	 if (&(*(iPair->val)) == &(*track)) {
	    traj = &(*(iPair->key));
	 }
      }

      // if good track, good trajectory
      if (track->pt() > 20.  &&  traj != NULL  &&  traj->isValid()) {

	 // calculate all residuals on this track
	 MuonResidualsFromTrack muonResidualsFromTrack(globalGeometry, traj, &(*track), m_muonAlignment->getAlignableNavigator(), 1000.);
	 std::vector<DetId> chamberIds = muonResidualsFromTrack.chamberIds();

	 // if the tracker part of refit is okay
	 if (muonResidualsFromTrack.trackerNumHits() >= 10  &&  muonResidualsFromTrack.trackerRedChi2() < 10.) {

	    // loop over ALL chambers
	    for (std::vector<DetId>::const_iterator chamberId = chamberIds.begin();  chamberId != chamberIds.end();  ++chamberId) {

	       // if CSC
	       if (chamberId->det() == DetId::Muon  &&  chamberId->subdetId() == MuonSubdetId::CSC) {

		  CSCDetId cscid(chamberId->rawId());
		  int station = (cscid.endcap() == 1 ? 1 : -1) * (10*cscid.station() + cscid.ring());
		  MuonChamberResidual *csc = muonResidualsFromTrack.chamberResidual(*chamberId, MuonChamberResidual::kCSC);

		  // if this segment is okay and has 6 hits
		  if (csc != NULL  &&  csc->numHits() >= 6) {
		     // fill the TTree
		     m_ttree_station = station;
		     m_ttree_chamber = cscid.chamber();
		     m_ttree_resid = csc->residual();
		     m_ttree_residslope = csc->resslope();
		     m_ttree_phi = csc->global_trackpos().phi();
		     m_ttree_qoverpt = double(track->charge()) / track->pt();
		     m_ttree->Fill();
		  } // end if CSC is okay

	       } // end if CSC

	    } // end loop over all chambers 

	 } // end if tracker part of refit is okay

      } // end if good track, good track refit

   } // end loop over tracks
}


// ------------ method called once each job just before starting event loop  ------------
void 
StandAloneTest::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
StandAloneTest::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(StandAloneTest);
