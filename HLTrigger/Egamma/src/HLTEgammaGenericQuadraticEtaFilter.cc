/** \class HLTEgammaGenericQuadraticEtaFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTrigger/Egamma/interface/HLTEgammaGenericQuadraticEtaFilter.h"

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
HLTEgammaGenericQuadraticEtaFilter::HLTEgammaGenericQuadraticEtaFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig){
  candTag_         = iConfig.getParameter< edm::InputTag > ("candTag");
  varTag_          = iConfig.getParameter< edm::InputTag > ("varTag");

  lessThan_        = iConfig.getParameter<bool> ("lessThan");
  useEt_           = iConfig.getParameter<bool> ("useEt");

  etaBoundaryEB12_ = iConfig.getParameter<double> ("etaBoundaryEB12");
  etaBoundaryEE12_ = iConfig.getParameter<double> ("etaBoundaryEE12");

  thrRegularEB1_   = iConfig.getParameter<double> ("thrRegularEB1");
  thrRegularEE1_   = iConfig.getParameter<double> ("thrRegularEE1");
  thrOverEEB1_     = iConfig.getParameter<double> ("thrOverEEB1");
  thrOverEEE1_     = iConfig.getParameter<double> ("thrOverEEE1");
  thrOverE2EB1_    = iConfig.getParameter<double> ("thrOverE2EB1");
  thrOverE2EE1_    = iConfig.getParameter<double> ("thrOverE2EE1");
  thrRegularEB2_   = iConfig.getParameter<double> ("thrRegularEB2");
  thrRegularEE2_   = iConfig.getParameter<double> ("thrRegularEE2");
  thrOverEEB2_     = iConfig.getParameter<double> ("thrOverEEB2");
  thrOverEEE2_     = iConfig.getParameter<double> ("thrOverEEE2");
  thrOverE2EB2_    = iConfig.getParameter<double> ("thrOverE2EB2");
  thrOverE2EE2_    = iConfig.getParameter<double> ("thrOverE2EE2");

  ncandcut_        = iConfig.getParameter<int> ("ncandcut");
			     				
  l1EGTag_         = iConfig.getParameter< edm::InputTag > ("l1EGCand");

  candToken_       = consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
  varToken_        = consumes<reco::RecoEcalCandidateIsolationMap> (varTag_);

//register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

void
HLTEgammaGenericQuadraticEtaFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltEGIsolFilter"));
  desc.add<edm::InputTag>("varTag", edm::InputTag("hltEGIsol"));
  desc.add<bool>("lessThan", true);
  desc.add<bool>("useEt", true);
  desc.add<double>("etaBoundaryEB12", 1.0);
  desc.add<double>("etaBoundaryEE12", 2.0);
  desc.add<double>("thrRegularEB1", 4.0);
  desc.add<double>("thrRegularEE1", 6.0);
  desc.add<double>("thrOverEEB1", 0.0020);
  desc.add<double>("thrOverEEE1", 0.0020);
  desc.add<double>("thrOverE2EB1", 0.0);
  desc.add<double>("thrOverE2EE1", 0.0);
  desc.add<double>("thrRegularEB2", 6.0);
  desc.add<double>("thrRegularEE2", 4.0);
  desc.add<double>("thrOverEEB2", 0.0020);
  desc.add<double>("thrOverEEE2", 0.0020);
  desc.add<double>("thrOverE2EB2", 0.0);
  desc.add<double>("thrOverE2EE2", 0.0);
  desc.add<int>("ncandcut", 1);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltEgammaGenericQuadraticEtaFilter", desc);
}

HLTEgammaGenericQuadraticEtaFilter::~HLTEgammaGenericQuadraticEtaFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaGenericQuadraticEtaFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;
  if ( saveTags() ) {
    filterproduct.addCollectionTag(l1EGTag_);
  }
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Set output format
  int trigger_type = trigger::TriggerCluster;
  if ( saveTags() ) trigger_type = trigger::TriggerPhoton;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByToken (candToken_,PrevFilterOutput);

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
      if (fabs(EtaSC) < etaBoundaryEB12_) {
          if ( vali <= thrRegularEB1_ + energy*thrOverEEB1_ + energy*energy*thrOverE2EB1_) {
	     n++;
	     filterproduct.addObject(trigger_type, ref);
	     continue;
          }
      } else if (fabs(EtaSC) < 1.479) {
          if ( vali <= thrRegularEB2_ + energy*thrOverEEB2_ + energy*energy*thrOverE2EB2_) {
	     n++;
	     filterproduct.addObject(trigger_type, ref);
	     continue;
          }
      } else if (fabs(EtaSC) < etaBoundaryEE12_) {
          if ( vali <= thrRegularEE1_ + energy*thrOverEEE1_ + energy*energy*thrOverE2EE1_) {
	    n++;
	    filterproduct.addObject(trigger_type, ref);
	    continue;
          }
      } else if (vali <= thrRegularEE2_ + energy*thrOverEEE2_ + energy*energy*thrOverE2EE2_) {
	  n++;
	  filterproduct.addObject(trigger_type, ref);
	  continue;
      }
    } else {
      if (fabs(EtaSC) < etaBoundaryEB12_) {
          if ( vali >= thrRegularEB1_ + energy*thrOverEEB1_ + energy*energy*thrOverE2EB1_) {
	     n++;
	     filterproduct.addObject(trigger_type, ref);
	     continue;
          }
      } else if (fabs(EtaSC) < 1.479) {
          if ( vali >= thrRegularEB2_ + energy*thrOverEEB2_ + energy*energy*thrOverE2EB2_) {
	     n++;
	     filterproduct.addObject(trigger_type, ref);
	     continue;
          }
      } else if (fabs(EtaSC) < etaBoundaryEE12_) {
          if ( vali >= thrRegularEE1_ + energy*thrOverEEE1_ + energy*energy*thrOverE2EE1_) {
	    n++;
	    filterproduct.addObject(trigger_type, ref);
	    continue;
          }
      } else if (vali >= thrRegularEE2_ + energy*thrOverEEE2_ + energy*energy*thrOverE2EE2_) {
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

