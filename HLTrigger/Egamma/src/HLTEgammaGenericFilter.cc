/** \class HLTEgammaGenericFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaGenericFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//
// constructors and destructor
//
HLTEgammaGenericFilter::HLTEgammaGenericFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");

  lessThan_ = iConfig.getParameter<bool> ("lessThan");			  
  useEt_ = iConfig.getParameter<bool> ("useEt");			  
  thrRegularEB_ = iConfig.getParameter<double> ("thrRegularEB");	  
  thrRegularEE_ = iConfig.getParameter<double> ("thrRegularEE");	  
  thrOverEEB_ = iConfig.getParameter<double> ("thrOverEEB");		  
  thrOverEEE_ = iConfig.getParameter<double> ("thrOverEEE");		  
  thrOverE2EB_ = iConfig.getParameter<double> ("thrOverE2EB");		  
  thrOverE2EE_ = iConfig.getParameter<double> ("thrOverE2EE");		  
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");			  
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");		  
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 	  
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  isoToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(isoTag_);
  if(!doIsolated_) nonIsoToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(nonIsoTag_);
}

void
HLTEgammaGenericFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag",edm::InputTag("hltSingleEgammaEtFilter"));
  desc.add<edm::InputTag>("isoTag",edm::InputTag("hltSingleEgammaHcalIsol"));
  desc.add<edm::InputTag>("nonIsoTag",edm::InputTag("hltSingleEgammaHcalNonIsol"));
  desc.add<bool>("lessThan",true);
  desc.add<bool>("useEt",false);
  desc.add<double>("thrRegularEB",0.0);
  desc.add<double>("thrRegularEE",0.0);
  desc.add<double>("thrOverEEB",-1.0);
  desc.add<double>("thrOverEEE",-1.0);
  desc.add<double>("thrOverE2EB",-1.0);
  desc.add<double>("thrOverE2EE",-1.0);
  desc.add<int>("ncandcut",1);
  desc.add<bool>("doIsolated",true);
  desc.add<edm::InputTag>("L1IsoCand",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<edm::InputTag>("L1NonIsoCand",edm::InputTag("hltL1NonIsoRecoEcalCandidate"));
  descriptions.add("hltEgammaGenericFilter",desc);
}

HLTEgammaGenericFilter::~HLTEgammaGenericFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaGenericFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (not doIsolated_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Set output format 
  int trigger_type = trigger::TriggerCluster;
  if (saveTags()) trigger_type = trigger::TriggerPhoton;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon
 
  //get hold of isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByToken (isoToken_,depMap);
  
  //get hold of non-isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByToken (nonIsoToken_,depNonIsoMap);
  
  // look at all photons, check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    
    ref = recoecalcands[i];
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );    
    if (mapi==(*depMap).end() && !doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
   
    float vali = mapi->val;
    float energy = ref->superCluster()->energy();
    float EtaSC = ref->eta();
    if (useEt_) energy = energy * sin (2*atan(exp(-EtaSC)));   
    
    if ( lessThan_ ) {
      if ( (fabs(EtaSC) < 1.479 && vali <= thrRegularEB_) || (fabs(EtaSC) >= 1.479 && vali <= thrRegularEE_) ) {
	n++;
	filterproduct.addObject(trigger_type, ref);
	continue;
      }
      if (energy > 0. && (thrOverEEB_ > 0. || thrOverEEE_ > 0. || thrOverE2EB_ > 0. || thrOverE2EE_ > 0.) ) {
	if ((fabs(EtaSC) < 1.479 && vali/energy <= thrOverEEB_) || (fabs(EtaSC) >= 1.479 && vali/energy <= thrOverEEE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	  continue;
	}
	if ((fabs(EtaSC) < 1.479 && vali/(energy*energy) <= thrOverE2EB_) || (fabs(EtaSC) >= 1.479 && vali/(energy*energy) <= thrOverE2EE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	}
      }
    } else {
      if ( (fabs(EtaSC) < 1.479 && vali >= thrRegularEB_) || (fabs(EtaSC) >= 1.479 && vali >= thrRegularEE_) ) {
	n++;
	filterproduct.addObject(trigger_type, ref);
	continue;
      }
      if (energy > 0. && (thrOverEEB_ > 0. || thrOverEEE_ > 0. || thrOverE2EB_ > 0. || thrOverE2EE_ > 0.) ) {
	if ((fabs(EtaSC) < 1.479 && vali/energy >= thrOverEEB_) || (fabs(EtaSC) >= 1.479 && vali/energy >= thrOverEEE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	  continue;
	}
	if ((fabs(EtaSC) < 1.479 && vali/(energy*energy) >= thrOverE2EB_) || (fabs(EtaSC) >= 1.479 && vali/(energy*energy) >= thrOverE2EE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	}
      }
    }
  }
  
  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}

