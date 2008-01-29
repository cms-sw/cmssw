/** \class HLTElectronTrackIsolFilter
 *
 * $Id: HLTElectronTrackIsolFilter.cc,v 1.5 2007/08/28 01:11:46 ratnik Exp $ 
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronTrackIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"


//
// constructors and destructor
//
HLTElectronTrackIsolFilter::HLTElectronTrackIsolFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  pttrackisolcut_  = iConfig.getParameter<double> ("pttrackisolcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronTrackIsolFilter::~HLTElectronTrackIsolFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronTrackIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all electrons,  check cuts and add to filter object
  int n = 0;

  edm::RefToBase<reco::Candidate> candref; 
  
  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    candref = recoecalcands->getParticleRef(i);
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    reco::SuperClusterRef recr2 = recr->superCluster();
    
    for(reco::ElectronIsolationMap::const_iterator it = depMap->begin(); it != depMap->end(); it++){
      
      reco::ElectronRef theElectronRef =  it->key;   
      const reco::SuperClusterRef theClus = theElectronRef->superCluster();
      ref=edm::RefToBase<reco::Candidate>(theElectronRef);

      if(&(*recr2) ==  &(*theClus)) {

	float vali = it->val;
	
	if(vali <= pttrackisolcut_){
	  n++;
	  filterproduct->putParticle(ref);
	}
      }   
    }
    if(!doIsolated_) {
     for(reco::ElectronIsolationMap::const_iterator it = depNonIsoMap->begin(); it != depNonIsoMap->end(); it++){
      
      reco::ElectronRef theElectronRef =  it->key;   
      const reco::SuperClusterRef theClus = theElectronRef->superCluster();
      ref=edm::RefToBase<reco::Candidate>(theElectronRef);

      if(&(*recr2) ==  &(*theClus)) {

	float vali = it->val;
	
	if(vali <= pttrackisolcut_){
	  n++;
	  filterproduct->putParticle(ref);
	}
      }   
     }
    }
   
  }
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}

