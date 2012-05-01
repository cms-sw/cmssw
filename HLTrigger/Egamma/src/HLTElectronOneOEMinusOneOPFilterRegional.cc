/** \class HLTElectronOneOEMinusOneOPFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id: HLTElectronOneOEMinusOneOPFilterRegional.cc,v 1.9 2011/01/19 16:48:23 sharper Exp $
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronOneOEMinusOneOPFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//
// constructors and destructor
//
HLTElectronOneOEMinusOneOPFilterRegional::HLTElectronOneOEMinusOneOPFilterRegional(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   electronIsolatedProducer_ = iConfig.getParameter< edm::InputTag > ("electronIsolatedProducer");
   electronNonIsolatedProducer_ = iConfig.getParameter< edm::InputTag > ("electronNonIsolatedProducer");
   barrelcut_  = iConfig.getParameter<double> ("barrelcut");
   endcapcut_  = iConfig.getParameter<double> ("endcapcut");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   doIsolated_  = iConfig.getParameter<bool> ("doIsolated");

   store_ = iConfig.getParameter<bool>("saveTags") ;
   // L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
   // L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}


HLTElectronOneOEMinusOneOPFilterRegional::~HLTElectronOneOEMinusOneOPFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronOneOEMinusOneOPFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The filter object
  using namespace trigger;
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(electronIsolatedProducer_);}
    if( store_ && !doIsolated_){filterproduct->addCollectionTag(electronNonIsolatedProducer_);}  
    //will be a collection of Ref<reco::ElectronCollection> ref;
    
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);

  edm::Handle<reco::ElectronCollection> electronIsolatedHandle;
  iEvent.getByLabel(electronIsolatedProducer_,electronIsolatedHandle);

  edm::Handle<reco::ElectronCollection> electronNonIsolatedHandle;
  if(!doIsolated_) {
    iEvent.getByLabel(electronNonIsolatedProducer_,electronNonIsolatedHandle);
  }

 // look at all candidates,  check cuts and add to filter object
  int n(0);
  
    //loop over all the RecoCandidates from the previous filter, 
    // associate them with the corresponding Electron object 
    //(the matching is done checking the super clusters)
    // and put into the event a Ref to the Electron objects that passes the
    // selections  
 
  edm::RefToBase<reco::Candidate> candref;   
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    reco::SuperClusterRef recr2 = recoecalcands[i]->superCluster();
    //loop over the electrons to find the matching one
    for(reco::ElectronCollection::const_iterator iElectron = electronIsolatedHandle->begin(); iElectron != electronIsolatedHandle->end(); iElectron++){
      // ElectronRef is a Ref<reco::RecoEcalCandidateCollection>   
      reco::ElectronRef electronref(reco::ElectronRef(electronIsolatedHandle,iElectron - electronIsolatedHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();
      if(&(*recr2) ==  &(*theClus)) {
	
	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp = 
				 fabs((1/electronref->superCluster()->energy()) - (1/trackMom.R()));

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < barrelcut_) {
	    n++;
	    filterproduct->addObject(TriggerElectron, electronref);
	  }
	}
	if( fabs(electronref->eta()) > 1.5 ) {
	  if ( elecEoverp < endcapcut_) {
	    n++;
	    filterproduct->addObject(TriggerElectron, electronref);
	  }
	}
      }//end of the if checking the matching of the SC from RecoCandidate and the one from Electrons
    }//end of loop over electrons

    if(!doIsolated_) {
    //loop over the electrons to find the matching one
    for(reco::ElectronCollection::const_iterator iElectron = electronNonIsolatedHandle->begin(); iElectron != electronNonIsolatedHandle->end(); iElectron++){
      
      reco::ElectronRef electronref(reco::ElectronRef(electronNonIsolatedHandle,iElectron - electronNonIsolatedHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();
      
      if(&(*recr2) ==  &(*theClus)) {

	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp = 
				fabs((1/electronref->superCluster()->energy()) - (1/trackMom.R())); 

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < barrelcut_) {
	    n++;
	    filterproduct->addObject(TriggerElectron, electronref);
	  }
	}
	if( fabs(electronref->eta()) > 1.5 ){
	  if ( elecEoverp < endcapcut_) {
	    n++;
	    filterproduct->addObject(TriggerElectron, electronref);
	  }
	}
      }//end of the if checking the matching of the SC from RecoCandidate and the one from Electrons
    }//end of loop over electrons
    }
  }//end of loop ober candidates

  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
