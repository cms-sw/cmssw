// -*- C++ -*-
//
// Package:    TrgMatchedMuonRefProducer
// Class:      TrgMatchedMuonRefProducer
// 
/**\class TrgMatchedMuonRefProducer TrgMatchedMuonRefProducer.cc PhysicsTools/TrgMatchedMuonRefProducer/src/TrgMatchedMuonRefProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Authors:  Nadia Adam, Valerie Halyo
//         Created:  Wed Oct  8 11:06:35 CDT 2008
// $Id: TrgMatchedMuonRefProducer.cc,v 1.2 2009/01/21 21:01:59 neadam Exp $
//
//

#include "PhysicsTools/TagAndProbe/interface/TrgMatchedMuonRefProducer.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h" 
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h" 
#include "DataFormats/Candidate/interface/Candidate.h" 
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/HLTPathStatus.h" 
#include "DataFormats/Common/interface/RefToBase.h" 
#include "DataFormats/Common/interface/TriggerResults.h" 
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
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
TrgMatchedMuonRefProducer::TrgMatchedMuonRefProducer(const edm::ParameterSet& iConfig)
{

   using namespace edm;
   using namespace reco;
   using namespace std;

   // Probe collection
   probeCollection_ = iConfig.getParameter<edm::InputTag>("ProbeCollection");

   // Trigger tags
   const edm::InputTag dTriggerEventTag("triggerSummaryRAW");
   triggerEventTag_ = 
      iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag",dTriggerEventTag);

   // Lepton filter tags
   vector<edm::InputTag> dMuonFilterTags;
   dMuonFilterTags.push_back(InputTag("hltSingleMuNoIsoL3PreFiltered9","","HLT"));
   muonFilterTags_ = iConfig.getUntrackedParameter< vector<edm::InputTag> >("muonFilterTags",dMuonFilterTags);

   // Matching cuts
   delRMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelRMatch",0.15);
   delPtRelMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelPtRelMatch",0.25);

   usePtMatching_ = iConfig.getUntrackedParameter<bool>("usePtMatching",true);

   produces<MuonRefVector>();
  
}


TrgMatchedMuonRefProducer::~TrgMatchedMuonRefProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrgMatchedMuonRefProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   using namespace std;
   using namespace l1extra; 
   using namespace trigger; 


   // We need the output Muon association collection to fill
   auto_ptr<MuonRefVector> muonTrgMatchedCollection( new MuonRefVector );


   // Read in the probe 

   Handle<  edm::RefVector<MuonCollection> > probes;
   if( !iEvent.getByLabel( probeCollection_, probes ) )
   {
      LogWarning("TrgMatch") << "Could not extract probe muons with input tag "
				 << probeCollection_;
   }

   // Trigger Info
   Handle<TriggerEvent> trgEvent;
   if( !iEvent.getByLabel(triggerEventTag_,trgEvent) )
   {
      LogWarning("TrgMatch") << "Could not extract trigger event summary "
			     << "with tag " << triggerEventTag_;
   }
  

   // Loop over Probes and trg match with Trigger objects
   if( trgEvent.isValid() &&  probes.isValid() )
   {       
      for( unsigned iProbe=0; iProbe<probes->size(); ++iProbe )
      {	  
	 const edm::Ref< MuonCollection > probeRef = (*probes)[iProbe];

	 reco::CandidateBaseRef probeBaseRef( probeRef );

	 double eta = probeRef->eta();
	 double phi = probeRef->phi();
	 double pt  = probeRef->pt();

	 //See if this muon fired the trigger/s in question
	 bool firedTrigger = false;

	 // loop over these objects to see whether they match
	 const trigger::TriggerObjectCollection& TOC = trgEvent->getObjects();
	       
	 for( int iF=0; iF<(int)muonFilterTags_.size(); ++iF )
	 {
	    size_type index = trgEvent->filterIndex( muonFilterTags_[iF] );
	    if( index >= trgEvent->sizeFilters() ) continue;

	    const trigger::Keys& KEYS(trgEvent->filterKeys(index));
	    for(int ipart = 0; ipart < (int)KEYS.size(); ++ipart) 
	    {  
	       const trigger::TriggerObject& TO = TOC[KEYS[ipart]];	
	       double dRval = deltaR((float)eta, (float)phi, TO.eta(), TO.phi());
	       if( !usePtMatching_ )
	       {
		  if( dRval < delRMatchingCut_ ) firedTrigger = true;
	       }
	       else
	       {
		  double dPtRel = (pt-TO.pt())/pt;
		  if( dRval < delRMatchingCut_ && fabs(dPtRel) < delPtRelMatchingCut_ ) firedTrigger = true;
	       } 
	    }
	 }
	 if (firedTrigger) muonTrgMatchedCollection->push_back(probeRef);		      
      }
   }

   // Finally put the tag probe collection in the event
   iEvent.put( muonTrgMatchedCollection );
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrgMatchedMuonRefProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrgMatchedMuonRefProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE(TrgMatchedMuonRefProducer);
