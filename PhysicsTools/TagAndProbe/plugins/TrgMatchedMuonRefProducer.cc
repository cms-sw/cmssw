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
// $Id$
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

   // Probe collection
   probeCollection_ = iConfig.getParameter<edm::InputTag>("ProbeCollection");

   // Trigger tags
   const edm::InputTag dTriggerEventTag("triggerSummaryRAW");
   triggerEventTag_ = 
      iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag",dTriggerEventTag);

   const edm::InputTag dHLTL1Tag("SingleMuIsoLevel1Seed");
   hltL1Tag_ = iConfig.getUntrackedParameter<edm::InputTag>("hltL1Tag",dHLTL1Tag);

   const edm::InputTag dHLTTag("SingleMuIsoL3IsoFiltered");
   hltTag_ = iConfig.getUntrackedParameter<edm::InputTag>("hltTag",dHLTTag);

   // Matching cuts
   delRMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelRMatch",0.15);
   delPtRelMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelPtRelMatch",0.25);

   simpleMatching_ = iConfig.getUntrackedParameter<bool>("useSimpleMatching",true);
   doL1Matching_ = iConfig.getUntrackedParameter<bool>("doL1Matching",true);
   usePtMatching_ = iConfig.getUntrackedParameter<bool>("usePtMatching",false);

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
  

   // Loop over Probes and trg match with L1/HLT objects
   if( trgEvent.isValid() &&  probes.isValid() )
   {       
      for( unsigned iProbe=0; iProbe<probes->size(); ++iProbe )
      {
	  
	 const edm::Ref< MuonCollection > probeRef = (*probes)[iProbe];

	 reco::CandidateBaseRef probeBaseRef( probeRef );

	 // Trigger objects
	 const TriggerObjectCollection& toColl = trgEvent->getObjects();

	 if( simpleMatching_ )
	 {
	    bool trigger = false;
	    for( int iObj=0; iObj<(int)toColl.size(); ++iObj )
	    {
	       const TriggerObject& hltObj( toColl[iObj] );
	       if( MatchObjects( hltObj, probeBaseRef, false ) )
	       {
		  //cout << "Found match!!" << endl;
		  trigger = true;
		  break;
	       }
	    }
	    if (trigger) muonTrgMatchedCollection->push_back(probeRef);		      
	 }
	 else
	 {
	    // Did this tag cause a L1 and/or HLT trigger?
	    bool l1Trigger = false;
	    bool hltTrigger = false;
	 
	    Keys l1Keys;
	    Keys hltKeys;
	    size_type numFilters = trgEvent->sizeFilters();

	    //cout << "Num filters " << numFilters << endl;
	    for( int iFilter = 0; iFilter<numFilters; ++iFilter )
	    {
	       if( doL1Matching_ )
	       {
		  if( trgEvent->filterTag(iFilter).label() == hltL1Tag_.label() )
		  {
		     //cout << "Here in match L1 tag " << trgEvent->filterTag(iFilter).label() << endl;
		     l1Keys = trgEvent->filterKeys(iFilter);
		     // Loop over the objects and see if any are matched
		     //cout << "Num keys " << l1Keys.size() << endl;
		     for( int iObj=0; iObj<(int)l1Keys.size(); ++ iObj )
		     {
			const TriggerObject& l1Obj( toColl[l1Keys[iObj]] );
			if( MatchObjects( l1Obj, probeBaseRef, false ) )
			{
			   //cout << "Found L1 match!!" << endl;
			   l1Trigger = true;
			   break;
			}
		     }
		  }
	       }
	       else
	       {
		  l1Trigger = true;
	       }

	       //cout << "L1 trigger " << l1Trigger << endl;

	       if( trgEvent->filterTag(iFilter).label() == hltTag_.label() )
	       {
		  //cout << "Here in match HLT tag " << trgEvent->filterTag(iFilter).label() << endl;
		  hltKeys = trgEvent->filterKeys(iFilter);
		  // Loop over the objects and see if any are matched
		  for( int iObj=0; iObj<(int)hltKeys.size(); ++ iObj )
		  {
		     const TriggerObject& hltObj( toColl[hltKeys[iObj]] );
		     if( MatchObjects( hltObj, probeBaseRef, false ) )
		     {
			//cout << "Found HLT match!!" << endl;
			hltTrigger = true;
			break;
		     }
		  }
	       }
	    }
	 
	    if (l1Trigger && hltTrigger) muonTrgMatchedCollection->push_back(probeRef);		      
	 }
      }
   }

   // Finally put the tag probe collection in the event
   iEvent.put( muonTrgMatchedCollection );
 
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrgMatchedMuonRefProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrgMatchedMuonRefProducer::endJob() {
}


bool 
TrgMatchedMuonRefProducer::MatchObjects( const trigger::TriggerObject& hltObj, 
                                         const reco::CandidateBaseRef& tagObj,
                                         bool exact )
 {

   double tEta = tagObj->eta();
   double tPhi = tagObj->phi();
   double tPt  = tagObj->pt();
   double hEta = hltObj.eta();
   double hPhi = hltObj.phi();
   double hPt  = hltObj.pt();
   
   double dRval = deltaR(tEta, tPhi, hEta, hPhi);
   double dPtRel = 999.0;
   if( tPt > 0.0 ) dPtRel = fabs( hPt - tPt )/tPt;
   
   // If we are comparing two objects for which the candidates should
   // be exactly the same, cut hard. Otherwise take cuts from user.
   if( usePtMatching_ )
   {
      if( exact ) return ( dRval < 1e-3 && dPtRel < 1e-3 );
      else        return ( dRval < delRMatchingCut_ && dPtRel < delPtRelMatchingCut_ );
   }
   else
   {
      if( exact ) return ( dRval < 1e-3 );
      else        return ( dRval < delRMatchingCut_ );
   }
 }

//define this as a plug-in
DEFINE_FWK_MODULE(TrgMatchedMuonRefProducer);
