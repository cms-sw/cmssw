// -*- C++ -*-
//
// Package:    MuTrkMatchedRecoChargedCandRefProducer
// Class:      MuTrkMatchedRecoChargedCandRefProducer
// 
/**\class MuTrkMatchedRecoChargedCandRefProducer MuTrkMatchedRecoChargedCandRefProducer.cc PhysicsTools/MuTrkMatchedRecoChargedCandRefProducer/src/MuTrkMatchedRecoChargedCandRefProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Nadia Adam, Valerie Halyo
//         Created:  Wed Oct  8 11:06:35 CDT 2008
// $Id: MuTrkMatchedRecoChargedCandRefProducer.cc,v 1.2 2009/01/21 21:01:59 neadam Exp $
//
//

#include "PhysicsTools/TagAndProbe/interface/MuTrkMatchedRecoChargedCandRefProducer.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h" 
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/HLTPathStatus.h" 
#include "DataFormats/Common/interface/RefToBase.h" 
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Math/GenVector/VectorUtil.h"

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
MuTrkMatchedRecoChargedCandRefProducer::MuTrkMatchedRecoChargedCandRefProducer(const edm::ParameterSet& iConfig)
{

   using namespace edm;
   using namespace reco;
   using namespace std;

   // Muon collection
   muonCollection_ = iConfig.getParameter<edm::InputTag>("MuonCollection");

   // Track collection
   trackCollection_ = iConfig.getParameter<edm::InputTag>("TrackCollection");


   produces<RecoChargedCandidateRefVector>();
  
}


MuTrkMatchedRecoChargedCandRefProducer::~MuTrkMatchedRecoChargedCandRefProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuTrkMatchedRecoChargedCandRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;


   // We need the output Muon association collection to fill
   auto_ptr<RecoChargedCandidateRefVector> muTrkMatchedCollection( new RecoChargedCandidateRefVector );


   // Read in the muons
   Handle<  edm::RefVector<MuonCollection> > muons;
   if( !iEvent.getByLabel( muonCollection_, muons ) )
   {
      LogWarning("TrkMatch") << "Could not extract muons with input tag "
				 << muonCollection_;
   }

   // Get the tracks
   Handle<  edm::RefVector<RecoChargedCandidateCollection> > tracks;
   if( !iEvent.getByLabel( trackCollection_, tracks ) )
   {
      LogWarning("TrkMatch") << "Could not extract reco cand tracks with input tag "
				 << trackCollection_;
   }

   // Loop over the tracks, and see which ones are matched to muon-tracks
   if( tracks.isValid() &&  muons.isValid() ){
       
      for( unsigned iTrack=0; iTrack<tracks->size(); ++iTrack ){
	  
	 const edm::Ref< RecoChargedCandidateCollection > rcRef = (*tracks)[iTrack];
	 TrackRef trkRef = rcRef->track();

	 if( trkRef.isNull() ) continue;

	 bool fndMatch = false;
	 for( unsigned iMuon=0; iMuon<muons->size(); ++iMuon ){
	    const edm::Ref< MuonCollection > muRef = (*muons)[iMuon];
	    
	    TrackRef muTrkRef = muRef->innerTrack();
	    if( muTrkRef.isNull() ) continue;

	    if( muTrkRef == trkRef ) fndMatch = true;
	 }
	 if (fndMatch) muTrkMatchedCollection->push_back(rcRef);		      
      }
   }

   // Finally put the tag probe collection in the event
   //std::cout << "Trk: " << muTrkMatchedCollection->size() << std::endl;

   iEvent.put( muTrkMatchedCollection );
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuTrkMatchedRecoChargedCandRefProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuTrkMatchedRecoChargedCandRefProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(MuTrkMatchedRecoChargedCandRefProducer);
