/** \class HLTEgammaGenericQuadraticFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTEgammaGenericQuadraticFilter.h"

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
HLTEgammaGenericQuadraticFilter::HLTEgammaGenericQuadraticFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter<edm::InputTag>("candTag");
  varTag_ = iConfig.getParameter<edm::InputTag>("varTag");
  rhoTag_ = iConfig.getParameter<edm::InputTag>("rhoTag");

  energyLowEdges_ = iConfig.getParameter<std::vector<double> >("energyLowEdges");
  lessThan_ = iConfig.getParameter<bool>("lessThan");
  useEt_ = iConfig.getParameter<bool>("useEt");

  thrRegularEB_ = iConfig.getParameter<std::vector<double> >("thrRegularEB");
  thrRegularEE_ = iConfig.getParameter<std::vector<double> >("thrRegularEE");
  thrOverEEB_ = iConfig.getParameter<std::vector<double> >("thrOverEEB");
  thrOverEEE_ = iConfig.getParameter<std::vector<double> >("thrOverEEE");
  thrOverE2EB_ = iConfig.getParameter<std::vector<double> >("thrOverE2EB");
  thrOverE2EE_ = iConfig.getParameter<std::vector<double> >("thrOverE2EE");

  ncandcut_ = iConfig.getParameter<int>("ncandcut");

  doRhoCorrection_ = iConfig.getParameter<bool>("doRhoCorrection");
  rhoMax_ = iConfig.getParameter<double>("rhoMax");
  rhoScale_ = iConfig.getParameter<double>("rhoScale");
  effectiveAreas_ = iConfig.getParameter<std::vector<double> >("effectiveAreas");
  absEtaLowEdges_ = iConfig.getParameter<std::vector<double> >("absEtaLowEdges");

  l1EGTag_ = iConfig.getParameter<edm::InputTag>("l1EGCand");

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  varToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(varTag_);

  if (energyLowEdges_.size() != thrRegularEB_.size() or energyLowEdges_.size() != thrRegularEE_.size() or
      energyLowEdges_.size() != thrOverEEB_.size() or energyLowEdges_.size() != thrOverEEE_.size() or
      energyLowEdges_.size() != thrOverE2EB_.size() or energyLowEdges_.size() != thrOverE2EE_.size())
    throw cms::Exception("IncompatibleVects") << "energyLowEdges and threshold vectors should be of the same size. \n";

  if (energyLowEdges_.at(0) != 0.0)
    throw cms::Exception("IncompleteCoverage") << "energyLowEdges should start from 0. \n";

  for (unsigned int aIt = 0; aIt < energyLowEdges_.size() - 1; aIt++) {
    if (!(energyLowEdges_.at(aIt) < energyLowEdges_.at(aIt + 1)))
      throw cms::Exception("ImproperBinning") << "energyLowEdges entries should be in increasing order. \n";
  }

  if (doRhoCorrection_) {
    rhoToken_ = consumes<double>(rhoTag_);
    if (absEtaLowEdges_.size() != effectiveAreas_.size())
      throw cms::Exception("IncompatibleVects") << "absEtaLowEdges and effectiveAreas should be of the same size. \n";

    if (absEtaLowEdges_.at(0) != 0.0)
      throw cms::Exception("IncompleteCoverage") << "absEtaLowEdges should start from 0. \n";

    for (unsigned int bIt = 0; bIt < absEtaLowEdges_.size() - 1; bIt++) {
      if (!(absEtaLowEdges_.at(bIt) < absEtaLowEdges_.at(bIt + 1)))
        throw cms::Exception("ImproperBinning") << "absEtaLowEdges entries should be in increasing order. \n";
    }
  }
}

void HLTEgammaGenericQuadraticFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltSingleEgammaEtFilter"));
  desc.add<edm::InputTag>("varTag", edm::InputTag("hltSingleEgammaHcalIsol"));
  desc.add<edm::InputTag>("rhoTag", edm::InputTag(""));     // No rho correction by default
  desc.add<std::vector<double> >("energyLowEdges", {0.0});  // No energy-dependent cuts by default
  desc.add<bool>("lessThan", true);
  desc.add<bool>("useEt", false);
  desc.add<std::vector<double> >("thrRegularEB", {0.0});
  desc.add<std::vector<double> >("thrRegularEE", {0.0});
  desc.add<std::vector<double> >("thrOverEEB", {-1.0});
  desc.add<std::vector<double> >("thrOverEEE", {-1.0});
  desc.add<std::vector<double> >("thrOverE2EB", {-1.0});
  desc.add<std::vector<double> >("thrOverE2EE", {-1.0});
  desc.add<int>("ncandcut", 1);
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 9.9999999E7);
  desc.add<double>("rhoScale", 1.0);
  desc.add<std::vector<double> >("effectiveAreas", {0.0, 0.0});
  desc.add<std::vector<double> >("absEtaLowEdges", {0.0, 1.479});  // EB, EE
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltEgammaGenericQuadraticFilter", desc);
}

HLTEgammaGenericQuadraticFilter::~HLTEgammaGenericQuadraticFilter() {}

// ------------ method called to produce the data  ------------
bool HLTEgammaGenericQuadraticFilter::hltFilter(edm::Event& iEvent,
                                                const edm::EventSetup& iSetup,
                                                trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Set output format
  int trigger_type = trigger::TriggerCluster;
  if (saveTags())
    trigger_type = trigger::TriggerPhoton;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(candToken_, PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if (recoecalcands.empty())
    PrevFilterOutput->getObjects(TriggerPhoton,
                                 recoecalcands);  //we dont know if its type trigger cluster or trigger photon

  //get hold of isolated association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByToken(varToken_, depMap);

  // Get rho if needed
  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoToken_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;
  rho = rho * rhoScale_;

  // look at all photons, check cuts and add to filter object
  int n = 0;
  for (unsigned int i = 0; i < recoecalcands.size(); i++) {
    ref = recoecalcands[i];
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find(ref);

    float vali = mapi->val;
    float EtaSC = ref->eta();

    // Pick the right EA and do rhoCorr
    if (doRhoCorrection_) {
      auto cIt = std::lower_bound(absEtaLowEdges_.begin(), absEtaLowEdges_.end(), std::abs(EtaSC)) - 1;
      vali = vali - (rho * effectiveAreas_.at(std::distance(absEtaLowEdges_.begin(), cIt)));
    }

    float energy = ref->superCluster()->energy();
    if (useEt_)
      energy = energy * sin(2 * atan(exp(-EtaSC)));
    if (energy < 0.)
      energy = 0.; /* first and second order terms assume non-negative energies */

    // Pick the right cut threshold
    double cutRegularEB_ = 9999., cutRegularEE_ = 9999.;
    double cutOverEEB_ = 9999., cutOverEEE_ = 9999.;
    double cutOverE2EB_ = 9999., cutOverE2EE_ = 9999.;

    auto dIt = std::lower_bound(energyLowEdges_.begin(), energyLowEdges_.end(), energy) - 1;
    unsigned iEn = std::distance(energyLowEdges_.begin(), dIt);

    cutRegularEB_ = thrRegularEB_.at(iEn);
    cutRegularEE_ = thrRegularEE_.at(iEn);
    cutOverEEB_ = thrOverEEB_.at(iEn);
    cutOverEEE_ = thrOverEEE_.at(iEn);
    cutOverE2EB_ = thrOverE2EB_.at(iEn);
    cutOverE2EE_ = thrOverE2EE_.at(iEn);

    if (lessThan_) {
      if ((std::abs(EtaSC) < 1.479 && vali <= cutRegularEB_ + energy * cutOverEEB_ + energy * energy * cutOverE2EB_) ||
          (std::abs(EtaSC) >= 1.479 && vali <= cutRegularEE_ + energy * cutOverEEE_ + energy * energy * cutOverE2EE_)) {
        n++;
        filterproduct.addObject(trigger_type, ref);
        continue;
      }
    } else {
      if ((std::abs(EtaSC) < 1.479 && vali >= cutRegularEB_ + energy * cutOverEEB_ + energy * energy * cutOverE2EB_) ||
          (std::abs(EtaSC) >= 1.479 && vali >= cutRegularEE_ + energy * cutOverEEE_ + energy * energy * cutOverE2EE_)) {
        n++;
        filterproduct.addObject(trigger_type, ref);
        continue;
      }
    }
  }

  // filter decision
  bool accept(n >= ncandcut_);

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaGenericQuadraticFilter);
