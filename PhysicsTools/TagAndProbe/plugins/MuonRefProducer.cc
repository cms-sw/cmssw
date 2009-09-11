// -*- C++ -*-
//
// Package:    MuonRefProducer
// Class:      MuonRefProducer
// 
/**\class MuonRefProducer MuonRefProducer.cc PhysicsTools/MuonRefProducer/src/MuonRefProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Nadia Adam, Valerie Halyo
//         Created:  Wed Oct  8 11:06:35 CDT 2008
// $Id: MuonRefProducer.cc,v 1.1 2008/10/13 20:45:39 neadam Exp $
//
//

#include "PhysicsTools/TagAndProbe/interface/MuonRefProducer.h"
#include "DataFormats/Candidate/interface/Candidate.h" 
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/HLTPathStatus.h" 
#include "DataFormats/Common/interface/RefToBase.h" 
#include "DataFormats/Common/interface/TriggerResults.h" 
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"


//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
MuonRefProducer::MuonRefProducer(const edm::ParameterSet& iConfig)
{

   // Probe collection
   probeCollection_ = iConfig.getParameter<edm::InputTag>("InputMuonCollection");

   // Matching cuts
   ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",3.0);
   useCharge_ = iConfig.getUntrackedParameter<bool>("useCharge",false);
   charge_ = iConfig.getUntrackedParameter<int>("charge",-1);
   nhits_ = iConfig.getUntrackedParameter<int>("nhits",11);
   nchi2_ = iConfig.getUntrackedParameter<double>("nchi2",10.0);
   d0_ = iConfig.getUntrackedParameter<double>("d0",2.0);
   z0_ = iConfig.getUntrackedParameter<double>("z0",25.0);

   produces<reco::MuonRefVector>();
  
}


MuonRefProducer::~MuonRefProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{



   // We need the output Muon association collection to fill
   std::auto_ptr<reco::MuonRefVector> muonCollection( new reco::MuonRefVector );


   // Read in the probe 



   edm::Handle<  edm::RefVector<reco::MuonCollection> > probes;
   if( !iEvent.getByLabel( probeCollection_, probes ) )
   {
      edm::LogWarning("MuonRefProducer") << "Could not extract probe muons with input tag "
				 << probeCollection_;
   }


   // Beam spot for muon tracks
   edm::Handle<reco::BeamSpot> beamSpot;
   if( !iEvent.getByLabel("offlineBeamSpot",beamSpot) )
   {
     edm::LogError("MuonRefProducer") << "Could not extract Beam Spot with input tag";
				
   }


   // Loop over Probes and find passing
   if( probes.isValid() )
   {       
      for( unsigned iProbe=0; iProbe<probes->size(); ++iProbe )
      {	  
	 const edm::Ref< reco::MuonCollection > probeRef = (*probes)[iProbe];

	 reco::CandidateBaseRef probeBaseRef( probeRef );

	 double pt  = probeRef->pt();
	 if( pt < ptMin_ ) continue;

	 if( useCharge_ && (probeRef->charge()*charge_ < 0) ) continue;

	 // Require tracker muon
	 if( !probeRef->isTrackerMuon() ) continue;

	 reco::TrackRef trkRef = probeRef->innerTrack();
	 if( trkRef.isNull() ) continue; //should not occur

	 // Require nhits track > 11
	 if( trkRef->found() <= nhits_ ) continue;

	 // Require nChi2 < 10.0
	 if( trkRef->normalizedChi2() > nchi2_ ) continue;

	 // Require d0(BS) < 2mm (0.2cm)
	 if( beamSpot.isValid() && fabs(trkRef->dxy( beamSpot->position() )) > d0_ ) continue;

	 // Require z0(BS) < 25cm 
	 if( beamSpot.isValid() && fabs(trkRef->dz( beamSpot->position() )) > z0_ ) continue;

	 // Require TMLastStationOptimizedLowPtTight is true
	 //if( !probeRef->isGood(reco::Muon::TMLastStationOptimizedLowPtTight) ) continue;
	 if( !probeRef->isGood(reco::Muon::TMOneStationTight) ) continue;

	 muonCollection->push_back(probeRef);		      
      }
   }


   // Finally put the tag probe collection in the event
   iEvent.put( muonCollection );
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonRefProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonRefProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(MuonRefProducer);
