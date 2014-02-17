/** \class HLTRapGapFilter
 *
 * $Id: HLTRapGapFilter.cc,v 1.12 2012/02/12 09:34:06 gruen Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTRapGapFilter::HLTRapGapFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
   inputTag_   = iConfig.getParameter< edm::InputTag > ("inputTag");
   absEtaMin_  = iConfig.getParameter<double> ("minEta");
   absEtaMax_  = iConfig.getParameter<double> ("maxEta"); 
   caloThresh_ = iConfig.getParameter<double> ("caloThresh"); 
}

HLTRapGapFilter::~HLTRapGapFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTRapGapFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace reco;
  using namespace trigger;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  edm::Handle<CaloJetCollection> recocalojets;
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

    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet!=(recocalojets->end()); recocalojet++) {
      
      etjet = recocalojet->energy();
      etajet = recocalojet->eta();
      
      if(std::abs(etajet) > absEtaMin_ && std::abs(etajet) < absEtaMax_)
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
      for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	   recocalojet!=(recocalojets->end()); recocalojet++) {
	CaloJetRef ref(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
	filterproduct.addObject(TriggerJet,ref);
	n++;
      }
    }
    
  } // events with two or more jets
  
  // filter decision
  bool accept(n>0);
  
  return accept;
}
