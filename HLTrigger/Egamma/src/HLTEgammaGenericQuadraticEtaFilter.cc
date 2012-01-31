/** \class HLTEgammaGenericQuadraticEtaFilter
 *
 * $Id: HLTEgammaGenericQuadraticEtaFilter.cc,v 1.2 2011/05/01 08:14:08 gruen Exp $
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/Egamma/interface/HLTEgammaGenericQuadraticEtaFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTEgammaGenericQuadraticEtaFilter::HLTEgammaGenericQuadraticEtaFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");

  lessThan_ = iConfig.getParameter<bool> ("lessThan");			  
  useEt_ = iConfig.getParameter<bool> ("useEt");			  
  etaBoundaryEB12_ = iConfig.getParameter<double> ("etaBoundaryEB12");	  
  etaBoundaryEE12_ = iConfig.getParameter<double> ("etaBoundaryEE12");	  
  thrRegularEB1_ = iConfig.getParameter<double> ("thrRegularEB1");	  
  thrRegularEE1_ = iConfig.getParameter<double> ("thrRegularEE1");	  
  thrOverEEB1_ = iConfig.getParameter<double> ("thrOverEEB1");		  
  thrOverEEE1_ = iConfig.getParameter<double> ("thrOverEEE1");		  
  thrOverE2EB1_ = iConfig.getParameter<double> ("thrOverE2EB1");		  
  thrOverE2EE1_ = iConfig.getParameter<double> ("thrOverE2EE1");		  
  thrRegularEB2_ = iConfig.getParameter<double> ("thrRegularEB2");	  
  thrRegularEE2_ = iConfig.getParameter<double> ("thrRegularEE2");	  
  thrOverEEB2_ = iConfig.getParameter<double> ("thrOverEEB2");		  
  thrOverEEE2_ = iConfig.getParameter<double> ("thrOverEEE2");		  
  thrOverE2EB2_ = iConfig.getParameter<double> ("thrOverE2EB2");		  
  thrOverE2EE2_ = iConfig.getParameter<double> ("thrOverE2EE2");		  
  				     	  
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");			  
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");		  
			     				  
  store_ = iConfig.getParameter<bool>("saveTags") ;	  
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 	  
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

//register your products
produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaGenericQuadraticEtaFilter::~HLTEgammaGenericQuadraticEtaFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaGenericQuadraticEtaFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Set output format 
  int trigger_type = trigger::TriggerCluster;
  if ( store_ ) trigger_type = trigger::TriggerPhoton;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
 
  //get hold of isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of non-isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);
  
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
    if (energy < 0.) energy=0.; /* first and second order terms assume non-negative energies */
    
    if ( lessThan_ ) {
      if (fabs(EtaSC) < etaBoundaryEB12_ && vali <= thrRegularEB1_ + energy*thrOverEEB1_ + energy*energy*thrOverE2EB1_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (fabs(EtaSC) < 1.479 && vali <= thrRegularEB2_ + energy*thrOverEEB2_ + energy*energy*thrOverE2EB2_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (fabs(EtaSC) < etaBoundaryEE12_ && vali <= thrRegularEE1_ + energy*thrOverEEE1_ + energy*energy*thrOverE2EE1_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (vali <= thrRegularEE2_ + energy*thrOverEEE2_ + energy*energy*thrOverE2EE2_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      }
    } else {
      if (fabs(EtaSC) < etaBoundaryEB12_ && vali >= thrRegularEB1_ + energy*thrOverEEB1_ + energy*energy*thrOverE2EB1_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (fabs(EtaSC) < 1.479 && vali >= thrRegularEB2_ + energy*thrOverEEB2_ + energy*energy*thrOverE2EB2_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (fabs(EtaSC) < etaBoundaryEE12_ && vali >= thrRegularEE1_ + energy*thrOverEEE1_ + energy*energy*thrOverE2EE1_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      } else if (vali >= thrRegularEE2_ + energy*thrOverEEE2_ + energy*energy*thrOverE2EE2_) {
	  n++;
	  filterproduct->addObject(trigger_type, ref);
	  continue;
      }
    }
  }
  
  // filter decision
  bool accept(n>=ncandcut_);

  // put filter object into the Event
  iEvent.put(filterproduct);

  return accept;
}

