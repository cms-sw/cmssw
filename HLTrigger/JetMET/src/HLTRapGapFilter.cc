/** \class HLTRapGapFilter
 *
 * $Id: HLTRapGapFilter.cc,v 1.9 2011/05/01 08:40:25 gruen Exp $
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
HLTRapGapFilter::HLTRapGapFilter(const edm::ParameterSet& iConfig)
{
   inputTag_   = iConfig.getParameter< edm::InputTag > ("inputTag");
   saveTags_    = iConfig.getParameter<bool>("saveTags");
   absEtaMin_  = iConfig.getParameter<double> ("minEta");
   absEtaMax_  = iConfig.getParameter<double> ("maxEta"); 
   caloThresh_ = iConfig.getParameter<double> ("caloThresh"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTRapGapFilter::~HLTRapGapFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTRapGapFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;
  using namespace trigger;

  // The filter object
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) filterobject->addCollectionTag(inputTag_);

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
      for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	   recocalojet!=(recocalojets->end()); recocalojet++) {
	CaloJetRef ref(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
	filterobject->addObject(TriggerJet,ref);
	n++;
      }
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>0);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}
