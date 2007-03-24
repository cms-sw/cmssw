/** \class HLTElectronEoverpFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id: HLTElectronEoverpFilterRegional.cc,v 1.4 2007/03/07 19:13:59 monicava Exp $
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronEoverpFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//
// constructors and destructor
//
HLTElectronEoverpFilterRegional::HLTElectronEoverpFilterRegional(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   electronProducer_ = iConfig.getParameter< edm::InputTag > ("electronProducer");
   eoverpbarrelcut_  = iConfig.getParameter<double> ("eoverpbarrelcut");
   eoverpendcapcut_  = iConfig.getParameter<double> ("eoverpendcapcut");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronEoverpFilterRegional::~HLTElectronEoverpFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronEoverpFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object (Electrons) to be recorded in filter object
  edm::RefToBase<reco::Candidate> outcandref; 


  // get hold of filtered candidates (RecoEcalCandidates)
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
   // Get the HLT electrons from EgammaHLTPixelMatchElectronProducers
  edm::Handle<reco::ElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_,electronHandle);

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
    for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
      
      reco::ElectronRef electronref(reco::ElectronRef(electronHandle,iElectron - electronHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();
      
      if(&(*recr2) ==  &(*theClus)) {

	outcandref=edm::RefToBase<reco::Candidate>(electronref);
	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp = 
				 electronref->superCluster()->energy()/ trackMom.R();

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < eoverpbarrelcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
	  }
	}
	if( (fabs(electronref->eta()) > 1.5) &&  (fabs(electronref->eta()) < 2.5) ){
	  if ( elecEoverp < eoverpendcapcut_) {
	    n++;
	    filterproduct->putParticle(outcandref);
	  }
	}
      }//end of the if checking the matching of the SC from RecoCandidate and the one from Electrons
    }//end of loop over electrons
  }//end of loop ober candidates

  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
