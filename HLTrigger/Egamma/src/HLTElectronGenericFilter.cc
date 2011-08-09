/** \class HLTElectronGenericFilter
 *
 * $Id: HLTElectronGenericFilter.cc,v 1.2 2009/09/16 09:23:20 covarell Exp $
 *
 *  \author Roberto Covarelli (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronGenericFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTElectronGenericFilter::HLTElectronGenericFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");

  lessThan_ = iConfig.getParameter<bool> ("lessThan");
  thrRegularEB_ = iConfig.getParameter<double> ("thrRegularEB");
  thrRegularEE_ = iConfig.getParameter<double> ("thrRegularEE");
  thrOverPtEB_ = iConfig.getParameter<double> ("thrOverPtEB");
  thrOverPtEE_ = iConfig.getParameter<double> ("thrOverPtEE");
  thrTimesPtEB_ = iConfig.getParameter<double> ("thrTimesPtEB");
  thrTimesPtEE_ = iConfig.getParameter<double> ("thrTimesPtEE");
  
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  store_ = iConfig.getParameter<bool>("saveTags") ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

//register your products
produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTElectronGenericFilter::~HLTElectronGenericFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronGenericFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}

  // Ref to Candidate object to be recorded in filter object
  reco::ElectronRef ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > elecands;
  PrevFilterOutput->getObjects(TriggerElectron, elecands);

  
  //get hold of isolated association map
  edm::Handle<reco::ElectronIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of non-isolated association map
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
  // look at all photons, check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<elecands.size(); i++) {
    
    ref = elecands[i];
    reco::ElectronIsolationMap::const_iterator mapi = (*depMap).find( ref );    
    if (mapi==(*depMap).end() && !doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
   
    float vali = mapi->val;
    float Pt = ref->pt();
    float Eta = fabs(ref->eta());

    if ( lessThan_ ) {
      if ( (Eta < 1.479 && vali <= thrRegularEB_) || (Eta >= 1.479 && vali <= thrRegularEE_) ) {
	n++;
	filterproduct->addObject(TriggerElectron, ref);
	continue;
      }
      if (Pt > 0. && (thrOverPtEB_ > 0. || thrOverPtEE_ > 0. || thrTimesPtEB_ > 0. || thrTimesPtEE_ > 0.) ) {
	if ((Eta < 1.479 && vali/Pt <= thrOverPtEB_) || (Eta >= 1.479 && vali/Pt <= thrOverPtEE_) ) {
	  n++;
	  filterproduct->addObject(TriggerElectron, ref);
	  continue;
	}
	if ((Eta < 1.479 && vali*Pt <= thrTimesPtEB_) || (Eta >= 1.479 && vali*Pt <= thrTimesPtEE_) ) {
	  n++;
	  filterproduct->addObject(TriggerElectron, ref);
	}
      }
    } else {
      if ( (Eta < 1.479 && vali >= thrRegularEB_) || (Eta >= 1.479 && vali >= thrRegularEE_) ) {
	n++;
	filterproduct->addObject(TriggerElectron, ref);
	continue;
      }
      if (Pt > 0. && (thrOverPtEB_ > 0. || thrOverPtEE_ > 0. || thrTimesPtEB_ > 0. || thrTimesPtEE_ > 0.) ) {
	if ((Eta < 1.479 && vali/Pt >= thrOverPtEB_) || (Eta >= 1.479 && vali/Pt >= thrOverPtEE_) ) {
	  n++;
	  filterproduct->addObject(TriggerElectron, ref);
	  continue;
	}
	if ((Eta < 1.479 && vali*Pt >= thrTimesPtEB_) || (Eta >= 1.479 && vali*Pt >= thrTimesPtEE_) ) {
	  n++;
	  filterproduct->addObject(TriggerElectron, ref);
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

