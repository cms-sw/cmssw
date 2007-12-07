/** \class HLTEgammaHOEFilter
 *
 * $Id: HLTEgammaHOEFilter.cc,v 1.2 2007/12/06 21:12:27 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 * identical to old HLTEgammaHcalIsolFilter but 
 *  -the Hcal et is devided by the supercluster et
 *  -the eta-range is not restricted to |eta|<2.5
 */

#include "HLTrigger/Egamma/interface/HLTEgammaHOEFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"


//
// constructors and destructor
//
HLTEgammaHOEFilter::HLTEgammaHOEFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  hcalisolbarrelcut_  = iConfig.getParameter<double> ("hcalisolbarrelcut");
  hcalisolendcapcut_  = iConfig.getParameter<double> ("hcalisolendcapcut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaHOEFilter::~HLTEgammaHOEFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaHOEFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerPhoton, recoecalcands);


  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);

  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    ref =  recoecalcands[i];   
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );
    
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
    }
     float vali = mapi->val / ref->et();
     
     if(fabs(ref->eta()) < 1.5){
       if ( vali < hcalisolbarrelcut_) {
	 n++;
	 filterproduct->addObject(TriggerPhoton, ref);
       }
     }
     if(
	(fabs(ref->eta()) > 1.5) //&& 
	){
       if ( vali < hcalisolendcapcut_) {
	 n++;
	 filterproduct->addObject(TriggerPhoton, ref);
       }
     }
  }
  
   // filter decision
   bool accept(n>=ncandcut_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   return accept;
}

