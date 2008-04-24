/** \class EgammaHLTEcalIsolFilter
 *
 * $Id: HLTEgammaEcalIsolFilter.cc,v 1.7 2007/12/07 14:41:33 ghezzi Exp $
 * 
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaEcalIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTEgammaEcalIsolFilter::HLTEgammaEcalIsolFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  ecalisolcut_  = iConfig.getParameter<double> ("ecalisolcut");
  ecalFracCut_ = iConfig.getParameter<double> ("ecalIsoOverEtCut");
  ecalIsoloEt2_ = iConfig.getParameter<double> ("ecalIsoOverEt2Cut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();

}

HLTEgammaEcalIsolFilter::~HLTEgammaEcalIsolFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaEcalIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The filter object
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  
  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);

  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of ecal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    ref = recoecalcands[i];
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );

    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
    }

    float vali = mapi->val;
    float IsoOE= vali/ref->et();
    float IsoOE2= IsoOE/ref->et();
    if ( vali < ecalisolcut_ || IsoOE < ecalFracCut_ || IsoOE2 < ecalIsoloEt2_ ) {
      n++;
      filterproduct->addObject(TriggerCluster, ref);
    }
   }
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

