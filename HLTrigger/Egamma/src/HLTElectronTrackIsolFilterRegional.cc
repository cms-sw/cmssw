/** \class HLTElectronTrackIsolFilterRegional
 *
 * $Id: HLTElectronTrackIsolFilterRegional.cc,v 1.1 2007/03/24 10:09:54 ghezzi Exp $ 
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronTrackIsolFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"


//
// constructors and destructor
//
HLTElectronTrackIsolFilterRegional::HLTElectronTrackIsolFilterRegional(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  pttrackisolcut_  = iConfig.getParameter<double> ("pttrackisolcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronTrackIsolFilterRegional::~HLTElectronTrackIsolFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronTrackIsolFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> elecands;
  iEvent.getByLabel (candTag_,elecands);
  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
    
  // look at all electrons,  check cuts and add to filter object
  int n = 0;

  edm::RefToBase<reco::Candidate> candref; 
  
  for (unsigned int i=0; i<elecands->size(); i++) {
    candref = elecands->getParticleRef(i);
    reco::ElectronRef eleref = candref.castTo<reco::ElectronRef>();
    
    reco::ElectronIsolationMap::const_iterator mapi = (*depMap).find( eleref );

    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( eleref ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }

    // Have to make sure that something is really found ????
    float vali = mapi->val;
    //for(reco::ElectronIsolationMap::const_iterator it = depMap->begin(); it != depMap->end(); it++){
    if(vali <= pttrackisolcut_){
      ref =edm::RefToBase<reco::Candidate>(eleref);
      n++;
      filterproduct->putParticle(ref);
    }
	
  }
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}

