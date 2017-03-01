/** \class HLTEgammaGenericQuadraticFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/Egamma/interface/HLTEgammaGenericQuadraticFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//
// constructors and destructor
//
HLTEgammaGenericQuadraticFilter::HLTEgammaGenericQuadraticFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  varTag_ = iConfig.getParameter< edm::InputTag > ("varTag");

  lessThan_ = iConfig.getParameter<bool> ("lessThan");
  useEt_ = iConfig.getParameter<bool> ("useEt");

  thrRegularEB_ = iConfig.getParameter<double> ("thrRegularEB");
  thrRegularEE_ = iConfig.getParameter<double> ("thrRegularEE");
  thrOverEEB_ = iConfig.getParameter<double> ("thrOverEEB");
  thrOverEEE_ = iConfig.getParameter<double> ("thrOverEEE");
  thrOverE2EB_ = iConfig.getParameter<double> ("thrOverE2EB");
  thrOverE2EE_ = iConfig.getParameter<double> ("thrOverE2EE");
  				     	
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
			     				
  l1EGTag_= iConfig.getParameter< edm::InputTag > ("l1EGCand");

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  varToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(varTag_);
}

void
HLTEgammaGenericQuadraticFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltSingleEgammaEtFilter"));
  desc.add<edm::InputTag>("varTag", edm::InputTag("hltSingleEgammaHcalIsol"));
  desc.add<bool>("lessThan", true);
  desc.add<bool>("useEt", false);
  desc.add<double>("thrRegularEB", 0.0);
  desc.add<double>("thrRegularEE", 0.0);
  desc.add<double>("thrOverEEB", 0.0);
  desc.add<double>("thrOverEEE", 0.0);
  desc.add<double>("thrOverE2EB", 0.0);
  desc.add<double>("thrOverE2EE", 0.0);
  desc.add<int>("ncandcut",1);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltEgammaGenericQuadraticFilter", desc);
}

HLTEgammaGenericQuadraticFilter::~HLTEgammaGenericQuadraticFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaGenericQuadraticFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Set output format
  int trigger_type = trigger::TriggerCluster;
  if (saveTags()) trigger_type = trigger::TriggerPhoton;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_, PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon

  //get hold of isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByToken (varToken_,depMap);

  // look at all photons, check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i];
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );

    float vali = mapi->val;
    float energy = ref->superCluster()->energy();
    float EtaSC = ref->eta();
    if (useEt_) energy = energy * sin (2*atan(exp(-EtaSC)));
    if (energy < 0.) energy=0.; /* first and second order terms assume non-negative energies */

    if ( lessThan_ ) {
      if ((fabs(EtaSC) < 1.479 && vali <= thrRegularEB_ + energy*thrOverEEB_ + energy*energy*thrOverE2EB_) || (fabs(EtaSC) >= 1.479 && vali <= thrRegularEE_ + energy*thrOverEEE_ + energy*energy*thrOverE2EE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	  continue;
      }
    } else {
      if ((fabs(EtaSC) < 1.479 && vali >= thrRegularEB_ + energy*thrOverEEB_ + energy*energy*thrOverE2EB_) || (fabs(EtaSC) >= 1.479 && vali >= thrRegularEE_ + energy*thrOverEEE_ + energy*energy*thrOverE2EE_) ) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	  continue;
      }
    }
  }

  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}
