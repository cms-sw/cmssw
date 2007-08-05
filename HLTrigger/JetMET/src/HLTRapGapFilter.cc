/** \class HLTRapGapFilter
 *
 * $Id: HLTRapGapFilter.cc,v 1.3 2007/08/01 12:22:45 elmer Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTRapGapFilter::HLTRapGapFilter(const edm::ParameterSet& iConfig)
{
   inputTag_ = iConfig.getParameter< edm::InputTag > ("inputTag");
   absEtaMin_   = iConfig.getParameter<double> ("minEta");
   absEtaMax_= iConfig.getParameter<double> ("maxEta"); 
   caloThresh_= iConfig.getParameter<double> ("caloThresh"); 

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTRapGapFilter::~HLTRapGapFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTRapGapFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  // Get the recoEcalCandidates

  edm::Handle<reco::CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);
  
  //std::cout << "Found " << recocalojets->size() << " jets in this event" << std::endl;

  if(recocalojets->size() > 1){
    // events with two or more jets

    double etjet=0.;
    double etajet=0.;
    double sumets=0.;
    int countjets =0;

    for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet!=(recocalojets->end()); recocalojet++) {
      
      etjet = recocalojet->energy();
      etajet = recocalojet->eta();
      
      if(fabs(etajet) > absEtaMin_ && fabs(etajet) < absEtaMax_)
	{
	  sumets += etjet;
	  //std::cout << "Adding jet with eta = " << etajet << ", and e = " 
	  //	    << etjet << std::endl;
	}
      countjets++;
    }

    //std::cout << "Sum jet energy = " << sumets << std::endl;
    if(sumets<=caloThresh_){
      //std::cout << "Passed filter!" << std::endl;
      for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	   recocalojet!=(recocalojets->end()); recocalojet++) {
	ref=edm::RefToBase<reco::Candidate>(reco::CaloJetRef(recocalojets,
							     distance(recocalojets->begin(),recocalojet)));
	filterproduct->putParticle(ref);
	n++;
      }
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=0);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
