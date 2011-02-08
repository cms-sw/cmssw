/** \class HLTCleanedJetVBFFilter
 *
 *  \author Andrea Benaglia
 *
 */

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "HLTrigger/JetMET/interface/HLTCleanedJetVBFFilter.h"


//! ctor
HLTCleanedJetVBFFilter::HLTCleanedJetVBFFilter(const edm::ParameterSet& iConfig):
  inputJetTag_   ( iConfig.getParameter<edm::InputTag>("inputJetTag") ),
  inputEleTag_   ( iConfig.getParameter<edm::InputTag>("inputEleTag") ),
  saveTag_       ( iConfig.getUntrackedParameter<bool>("saveTag",false) ),
  minCleaningDR_ ( iConfig.getParameter<double>("minCleaningDR") ),
  minJetEtHigh_  ( iConfig.getParameter<double>("minJetEtHigh") ),
  minJetEtLow_   ( iConfig.getParameter<double>("minJetEtLow") ),
  minJetDeta_    ( iConfig.getParameter<double>("minJetDeta") )
{    
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}



//! dtor
HLTCleanedJetVBFFilter::~HLTCleanedJetVBFFilter()
{}



//! the filter method
bool HLTCleanedJetVBFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject(new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( saveTag_ ) filterobject -> addCollectionTag(inputJetTag_);
  
  // get the ele trigger objects from previous filters
  edm::Handle<trigger::TriggerFilterObjectWithRefs> prevEleFilterOutput;
  iEvent.getByLabel(inputEleTag_, prevEleFilterOutput);
  
  // get the ele candidates from trigger objects
  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  prevEleFilterOutput->getObjects(trigger::TriggerElectron, eleCands);
  
  // get the jet candidates 
  edm::Handle<reco::CaloJetCollection> jetCands;
  iEvent.getByLabel(inputJetTag_, jetCands);
  
  
  
  // Check if there are two VBF jets after cleaning of ele candidates
  bool accept = false;
  
  // loop on ele candidates
  for(unsigned int eleIt = 0; eleIt < eleCands.size(); ++eleIt)
  {
    reco::ElectronRef eleRef = eleCands[eleIt];
    
    
    // loop on jet 1  
    for(reco::CaloJetCollection::const_iterator jetIt1 = jetCands->begin();
        jetIt1 != jetCands->end(); ++jetIt1)
    {
      // clean jet 1 from ele candidate 
      reco::CaloJetRef jetRef1(reco::CaloJetRef(jetCands,distance(jetCands->begin(),jetIt1)));
      if( deltaR(eleRef->eta(),eleRef->phi(),jetRef1->eta(),jetRef1->phi()) < minCleaningDR_ ) continue;
      
      
      // loop on jet 2
      for(reco::CaloJetCollection::const_iterator jetIt2 = jetIt1+1;
          jetIt2 != jetCands->end(); ++jetIt2)
      {
        // clean jet 2 from ele candidate
        reco::CaloJetRef jetRef2(reco::CaloJetRef(jetCands,distance(jetCands->begin(),jetIt2)));
        if( deltaR(eleRef->eta(),eleRef->phi(),jetRef2->eta(),jetRef2->phi()) < minCleaningDR_ ) continue;
        
        
        // asymmetric jet et cuts && Deta cut
        if( (std::max(jetRef1->et(),jetRef2->et()) > minJetEtHigh_) && 
            (std::min(jetRef1->et(),jetRef2->et()) > minJetEtLow_) && 
            (fabs(jetRef1->eta()-jetRef2->eta()) > minJetDeta_) )
        {
          accept = true;
	  filterobject->addObject(trigger::TriggerJet,jetRef1);
	  filterobject->addObject(trigger::TriggerJet,jetRef2);
        }

      } // loop on jet 2
    } // loop on jet 1
  } // loop on ele
  
  
  
  // put filter object into the Event
  if( saveTag_ )
    iEvent.put(filterobject);
  
  
  return accept;
}

// declare the class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTCleanedJetVBFFilter);
