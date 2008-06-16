// -*- C++ -*-
//
// Package:    ZeroFieldGlobalMuonBuilder
// Class:      ZeroFieldGlobalMuonBuilder
// 
/**\class ZeroFieldGlobalMuonBuilder ZeroFieldGlobalMuonBuilder.cc Alignment/ZeroFieldGlobalMuonBuilder/src/ZeroFieldGlobalMuonBuilder.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: ZeroFieldGlobalMuonBuilder.cc,v 1.8 2008/06/11 23:35:58 pivarski Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
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
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class ZeroFieldGlobalMuonBuilder : public edm::EDFilter {
   public:
      explicit ZeroFieldGlobalMuonBuilder(const edm::ParameterSet&);
      ~ZeroFieldGlobalMuonBuilder();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_inputMuon, m_inputTracker;
      int m_minMuonHits, m_minTrackerHits;
      double m_minPdot, m_minDdotP;
      bool m_debuggingHistograms;
      TH1F *th1f_muonHits, *th1f_trackerHits, *th1f_pdot, *th1f_ddotp, *th1f_displacement, *th1f_displacement2;

      unsigned long m_total_events, m_passing_cuts;
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
ZeroFieldGlobalMuonBuilder::ZeroFieldGlobalMuonBuilder(const edm::ParameterSet& iConfig)
   : m_inputMuon(iConfig.getParameter<edm::InputTag>("inputMuon"))
   , m_inputTracker(iConfig.getParameter<edm::InputTag>("inputTracker"))
   , m_minMuonHits(iConfig.getParameter<int>("minMuonHits"))
   , m_minTrackerHits(iConfig.getParameter<int>("minTrackerHits"))
   , m_minPdot(iConfig.getParameter<double>("minPdot"))
   , m_minDdotP(iConfig.getParameter<double>("minDdotP"))
   , m_debuggingHistograms(iConfig.getUntrackedParameter<bool>("debuggingHistograms", false))
{
   if (m_debuggingHistograms) {
      edm::Service<TFileService> tfile;
      th1f_muonHits = tfile->make<TH1F>("muonHits", "muonHits", 51, -0.5, 50.5);
      th1f_trackerHits = tfile->make<TH1F>("trackerHits", "trackerHits", 101, -0.5, 100.5);
      th1f_pdot = tfile->make<TH1F>("pdot", "pdot", 1000, 0.99, 1.);
      th1f_ddotp = tfile->make<TH1F>("ddotp", "ddotp", 1000, 0.99, 1.);
      th1f_displacement = tfile->make<TH1F>("displacement", "displacement", 300, 0., 300.);
      th1f_displacement2 = tfile->make<TH1F>("displacement2", "displacement2", 300, 0., 300.);
   }

   produces<reco::TrackCollection>();
   produces<reco::TrackExtraCollection>();
   produces<TrackingRecHitCollection>();

   m_total_events = 0;
   m_passing_cuts = 0;
}


ZeroFieldGlobalMuonBuilder::~ZeroFieldGlobalMuonBuilder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool
ZeroFieldGlobalMuonBuilder::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   m_total_events = 0;

   edm::Handle<reco::TrackCollection> tracksMuon;
   iEvent.getByLabel(m_inputMuon, tracksMuon);

   edm::Handle<reco::TrackCollection> tracksTracker;
   iEvent.getByLabel(m_inputTracker, tracksTracker);

   std::auto_ptr<reco::TrackCollection> trackCollection(new reco::TrackCollection);
   std::auto_ptr<reco::TrackExtraCollection> trackExtraCollection(new reco::TrackExtraCollection);
   std::auto_ptr<TrackingRecHitCollection> trackingRecHitCollection(new TrackingRecHitCollection);

   reco::TrackExtraRefProd refTrackExtra = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
   TrackingRecHitRefProd refTrackingRecHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
   edm::Ref<reco::TrackExtraCollection>::key_type refTrackExtraIndex = 0;
   edm::Ref<TrackingRecHitCollection>::key_type refTrackingRecHitsIndex = 0;

   int good_tracks = 0;
   for (reco::TrackCollection::const_iterator trackM = tracksMuon->begin();  trackM != tracksMuon->end();  ++trackM) {
      for (reco::TrackCollection::const_iterator trackT = tracksTracker->begin();  trackT != tracksTracker->end();  ++trackT) {
	 math::XYZVector momentumT = trackT->momentum();
	 math::XYZVector momentumM = trackM->momentum();
	 math::XYZVector displacement = trackT->referencePoint() - trackM->referencePoint();
	 
	 double momentum_dotproduct = fabs(momentumT.Dot(momentumM)) / sqrt(momentumT.Mag2()) / sqrt(momentumM.Mag2());
	 double displacement_dotproduct = fabs(displacement.Dot(momentumT)) / sqrt(displacement.Mag2()) / sqrt(momentumT.Mag2());

	 if (m_debuggingHistograms) {
	    th1f_muonHits->Fill(trackM->recHitsSize());
	    th1f_trackerHits->Fill(trackT->recHitsSize());
	    th1f_pdot->Fill(momentum_dotproduct);
	    th1f_ddotp->Fill(displacement_dotproduct);
	    th1f_displacement->Fill(sqrt(displacement.Mag2()));
	    if (displacement_dotproduct < m_minDdotP) {
	       th1f_displacement2->Fill(sqrt(displacement.Mag2()));
	    }
	 }

	 if (momentum_dotproduct > m_minPdot  &&  displacement_dotproduct > m_minDdotP) {
	    good_tracks++;

	    reco::Track *track = new reco::Track(trackT->chi2(), trackT->ndof(), trackT->referencePoint(), trackT->momentum(), trackT->charge(), trackT->covariance(), trackT->algo(), reco::TrackBase::loose);
	    reco::TrackExtra *trackExtra = new reco::TrackExtra(trackT->extra()->outerPosition(), trackT->extra()->outerMomentum(), trackT->extra()->outerOk(), trackT->extra()->innerPosition(), trackT->extra()->innerMomentum(), trackT->extra()->innerOk(), trackT->extra()->outerStateCovariance(), trackT->extra()->outerDetId(), trackT->extra()->innerStateCovariance(), trackT->extra()->innerDetId(), trackT->extra()->seedDirection(), trackT->extra()->seedRef());
	    track->setExtra(reco::TrackExtraRef(refTrackExtra, refTrackExtraIndex++));

	    for (trackingRecHit_iterator hit = trackT->recHitsBegin();  hit != trackT->recHitsEnd();  ++hit) {
	       TrackingRecHit *myhit = (*hit)->clone();
	       trackingRecHitCollection->push_back(myhit);
	       trackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	    }

	    for (trackingRecHit_iterator hit = trackM->recHitsBegin();  hit != trackM->recHitsEnd();  ++hit) {
	       TrackingRecHit *myhit = (*hit)->clone();
	       trackingRecHitCollection->push_back(myhit);
	       trackExtra->add(TrackingRecHitRef(refTrackingRecHits, refTrackingRecHitsIndex++));
	    }

	    trackCollection->push_back(*track);
	    trackExtraCollection->push_back(*trackExtra);
	 }
      }
   }

   iEvent.put(trackCollection);
   iEvent.put(trackExtraCollection);
   iEvent.put(trackingRecHitCollection);

   if (good_tracks > 0) {
      m_passing_cuts++;
      return true;
   }
   else return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
ZeroFieldGlobalMuonBuilder::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeroFieldGlobalMuonBuilder::endJob() {
   std::cout << "ZeroFieldGlobalMuonBuilder: total_events " << m_total_events << " passing_cuts " << m_passing_cuts << std::endl;

}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeroFieldGlobalMuonBuilder);
