/** \class HLTElectronOneOEMinusOneOPFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id: HLTElectronOneOEMinusOneOPFilterRegional.cc,v 1.1 2007/08/28 15:21:51 ghezzi Exp $
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronOneOEMinusOneOPFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

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


   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronOneOEMinusOneOPFilterRegional::~HLTElectronOneOEMinusOneOPFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronOneOEMinusOneOPFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object (Electrons) to be recorded in filter object
  edm::RefToBase<reco::Candidate> outcandref; 


  // get hold of filtered candidates (RecoEcalCandidates)
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
   // Get the HLT electrons from EgammaHLTPixelMatchElectronProducers
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
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    
    //reco::ElectronRef eleref = candref.castTo<reco::ElectronRef>();
    candref = recoecalcands->getParticleRef(i);
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    reco::SuperClusterRef recr2 = recr->superCluster();

    //loop over the electrons to find the matching one
    for(reco::ElectronCollection::const_iterator iElectron = electronIsolatedHandle->begin(); iElectron != electronIsolatedHandle->end(); iElectron++){
      
      reco::ElectronRef electronref(reco::ElectronRef(electronIsolatedHandle,iElectron - electronIsolatedHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();
      
      if(&(*recr2) ==  &(*theClus)) {

	outcandref=edm::RefToBase<reco::Candidate>(electronref);
	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp = 
				 fabs((1/electronref->superCluster()->energy()) - (1/trackMom.R()));

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < barrelcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
	  }
	}
	if( (fabs(electronref->eta()) > 1.5) &&  (fabs(electronref->eta()) < 2.5) ){
	  if ( elecEoverp < endcapcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
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

	outcandref=edm::RefToBase<reco::Candidate>(electronref);
	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp = 
				fabs((1/electronref->superCluster()->energy()) - (1/trackMom.R())); 

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < barrelcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
	  }
	}
	if( (fabs(electronref->eta()) > 1.5) &&  (fabs(electronref->eta()) < 2.5) ){
	  if ( elecEoverp < endcapcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
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
