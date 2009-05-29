/** \class HLTElectronTrackIsolFilterRegional
 *
 * $Id: HLTElectronTrackIsolFilterRegional.cc,v 1.7 2008/11/03 14:43:27 ghezzi Exp $ 
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronTrackIsolFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

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
  pttrackisolOverEcut_ = iConfig.getParameter<double> ("pttrackisolOverEcut");
  pttrackisolOverE2cut_ = iConfig.getParameter<double> ("pttrackisolOverE2cut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTElectronTrackIsolFilterRegional::~HLTElectronTrackIsolFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronTrackIsolFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
 // The filter object
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::ElectronCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > elecands;
  PrevFilterOutput->getObjects(TriggerElectron, elecands);

  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of track isolation association map
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
    
  // look at all electrons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<elecands.size(); i++) {

    reco::ElectronRef eleref = elecands[i];
    
    reco::ElectronIsolationMap::const_iterator mapi = (*depMap).find( eleref );

    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( eleref ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }

    // Have to make sure that something is really found ????
    float vali = mapi->val;//vali is (sum pt tracks)/pt_ele 

    //for(reco::ElectronIsolationMap::const_iterator it = depMap->begin(); it != depMap->end(); it++){
    if( (vali*eleref->pt() <= pttrackisolcut_) || (vali <=  pttrackisolOverEcut_) || (vali/eleref->pt() <= pttrackisolOverE2cut_) ){
      n++;
      filterproduct->addObject(TriggerElectron, eleref);
    }
	
  }
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}

