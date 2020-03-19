/** \class HLTEgammaGenericQuadraticEtaFilter
 *
 *
 *  \author Roberto Covarelli (CERN)
 *  modified by Chris Tully (Princeton)
 */

#include "HLTEgammaGenericQuadraticEtaFilter.h"

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
HLTEgammaGenericQuadraticEtaFilter::HLTEgammaGenericQuadraticEtaFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig) {
  candTag_ = iConfig.getParameter<edm::InputTag>("candTag");
  varTag_ = iConfig.getParameter<edm::InputTag>("varTag");
  rhoTag_ = iConfig.getParameter<edm::InputTag>("rhoTag");

  energyLowEdges_ = iConfig.getParameter<std::vector<double> >("energyLowEdges");
  lessThan_ = iConfig.getParameter<bool>("lessThan");
  useEt_ = iConfig.getParameter<bool>("useEt");

  etaBoundaryEB12_ = iConfig.getParameter<double>("etaBoundaryEB12");
  etaBoundaryEE12_ = iConfig.getParameter<double>("etaBoundaryEE12");

  thrRegularEB1_ = iConfig.getParameter<std::vector<double> >("thrRegularEB1");
  thrRegularEE1_ = iConfig.getParameter<std::vector<double> >("thrRegularEE1");
  thrOverEEB1_ = iConfig.getParameter<std::vector<double> >("thrOverEEB1");
  thrOverEEE1_ = iConfig.getParameter<std::vector<double> >("thrOverEEE1");
  thrOverE2EB1_ = iConfig.getParameter<std::vector<double> >("thrOverE2EB1");
  thrOverE2EE1_ = iConfig.getParameter<std::vector<double> >("thrOverE2EE1");
  thrRegularEB2_ = iConfig.getParameter<std::vector<double> >("thrRegularEB2");
  thrRegularEE2_ = iConfig.getParameter<std::vector<double> >("thrRegularEE2");
  thrOverEEB2_ = iConfig.getParameter<std::vector<double> >("thrOverEEB2");
  thrOverEEE2_ = iConfig.getParameter<std::vector<double> >("thrOverEEE2");
  thrOverE2EB2_ = iConfig.getParameter<std::vector<double> >("thrOverE2EB2");
  thrOverE2EE2_ = iConfig.getParameter<std::vector<double> >("thrOverE2EE2");

  ncandcut_ = iConfig.getParameter<int>("ncandcut");

  doRhoCorrection_ = iConfig.getParameter<bool>("doRhoCorrection");
  rhoMax_ = iConfig.getParameter<double>("rhoMax");
  rhoScale_ = iConfig.getParameter<double>("rhoScale");
  effectiveAreas_ = iConfig.getParameter<std::vector<double> >("effectiveAreas");
  absEtaLowEdges_ = iConfig.getParameter<std::vector<double> >("absEtaLowEdges");

  l1EGTag_ = iConfig.getParameter<edm::InputTag>("l1EGCand");

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  varToken_ = consumes<reco::RecoEcalCandidateIsolationMap>(varTag_);

  if (energyLowEdges_.size() != thrRegularEB1_.size() or energyLowEdges_.size() != thrRegularEE1_.size() or
      energyLowEdges_.size() != thrRegularEB2_.size() or energyLowEdges_.size() != thrRegularEE2_.size() or
      energyLowEdges_.size() != thrOverEEB1_.size() or energyLowEdges_.size() != thrOverEEE1_.size() or
      energyLowEdges_.size() != thrOverEEB2_.size() or energyLowEdges_.size() != thrOverEEE2_.size() or
      energyLowEdges_.size() != thrOverE2EB1_.size() or energyLowEdges_.size() != thrOverE2EE1_.size() or
      energyLowEdges_.size() != thrOverE2EB2_.size() or energyLowEdges_.size() != thrOverE2EE2_.size())
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

void HLTEgammaGenericQuadraticEtaFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltEGIsolFilter"));
  desc.add<edm::InputTag>("varTag", edm::InputTag("hltEGIsol"));
  desc.add<edm::InputTag>("rhoTag", edm::InputTag(""));     // No rho correction by default
  desc.add<std::vector<double> >("energyLowEdges", {0.0});  // No energy-dependent cuts by default
  desc.add<bool>("lessThan", true);
  desc.add<bool>("useEt", true);
  desc.add<double>("etaBoundaryEB12", 1.0);
  desc.add<double>("etaBoundaryEE12", 2.0);
  desc.add<std::vector<double> >("thrRegularEB1", {4.0});
  desc.add<std::vector<double> >("thrRegularEE1", {6.0});
  desc.add<std::vector<double> >("thrOverEEB1", {0.0020});
  desc.add<std::vector<double> >("thrOverEEE1", {0.0020});
  desc.add<std::vector<double> >("thrOverE2EB1", {0.0});
  desc.add<std::vector<double> >("thrOverE2EE1", {0.0});
  desc.add<std::vector<double> >("thrRegularEB2", {6.0});
  desc.add<std::vector<double> >("thrRegularEE2", {4.0});
  desc.add<std::vector<double> >("thrOverEEB2", {0.0020});
  desc.add<std::vector<double> >("thrOverEEE2", {0.0020});
  desc.add<std::vector<double> >("thrOverE2EB2", {0.0});
  desc.add<std::vector<double> >("thrOverE2EE2", {0.0});
  desc.add<int>("ncandcut", 1);
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 9.9999999E7);
  desc.add<double>("rhoScale", 1.0);
  desc.add<std::vector<double> >("effectiveAreas", {0.0, 0.0});
  desc.add<std::vector<double> >("absEtaLowEdges", {0.0, 1.479});  // EB, EE
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  descriptions.add("hltEgammaGenericQuadraticEtaFilter", desc);
}

HLTEgammaGenericQuadraticEtaFilter::~HLTEgammaGenericQuadraticEtaFilter() {}

// ------------ method called to produce the data  ------------
bool HLTEgammaGenericQuadraticEtaFilter::hltFilter(edm::Event& iEvent,
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

    double cutRegularEB1_ = 9999., cutRegularEE1_ = 9999.;
    double cutRegularEB2_ = 9999., cutRegularEE2_ = 9999.;
    double cutOverEEB1_ = 9999., cutOverEEE1_ = 9999.;
    double cutOverEEB2_ = 9999., cutOverEEE2_ = 9999.;
    double cutOverE2EB1_ = 9999., cutOverE2EE1_ = 9999.;
    double cutOverE2EB2_ = 9999., cutOverE2EE2_ = 9999.;

    auto dIt = std::lower_bound(energyLowEdges_.begin(), energyLowEdges_.end(), energy) - 1;
    unsigned iEn = std::distance(energyLowEdges_.begin(), dIt);

    cutRegularEB1_ = thrRegularEB1_.at(iEn);
    cutRegularEB2_ = thrRegularEB2_.at(iEn);
    cutRegularEE1_ = thrRegularEE1_.at(iEn);
    cutRegularEE2_ = thrRegularEE2_.at(iEn);
    cutOverEEB1_ = thrOverEEB1_.at(iEn);
    cutOverEEB2_ = thrOverEEB2_.at(iEn);
    cutOverEEE1_ = thrOverEEE1_.at(iEn);
    cutOverEEE2_ = thrOverEEE2_.at(iEn);
    cutOverE2EB1_ = thrOverE2EB1_.at(iEn);
    cutOverE2EB2_ = thrOverE2EB2_.at(iEn);
    cutOverE2EE1_ = thrOverE2EE1_.at(iEn);
    cutOverE2EE2_ = thrOverE2EE2_.at(iEn);

    if (lessThan_) {
      if (std::abs(EtaSC) < etaBoundaryEB12_) {
        if (vali <= cutRegularEB1_ + energy * cutOverEEB1_ + energy * energy * cutOverE2EB1_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (std::abs(EtaSC) < 1.479) {
        if (vali <= cutRegularEB2_ + energy * cutOverEEB2_ + energy * energy * cutOverE2EB2_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (std::abs(EtaSC) < etaBoundaryEE12_) {
        if (vali <= cutRegularEE1_ + energy * cutOverEEE1_ + energy * energy * cutOverE2EE1_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (vali <= cutRegularEE2_ + energy * cutOverEEE2_ + energy * energy * cutOverE2EE2_) {
        n++;
        filterproduct.addObject(trigger_type, ref);
        continue;
      }
    } else {
      if (std::abs(EtaSC) < etaBoundaryEB12_) {
        if (vali >= cutRegularEB1_ + energy * cutOverEEB1_ + energy * energy * cutOverE2EB1_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (std::abs(EtaSC) < 1.479) {
        if (vali >= cutRegularEB2_ + energy * cutOverEEB2_ + energy * energy * cutOverE2EB2_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (std::abs(EtaSC) < etaBoundaryEE12_) {
        if (vali >= cutRegularEE1_ + energy * cutOverEEE1_ + energy * energy * cutOverE2EE1_) {
          n++;
          filterproduct.addObject(trigger_type, ref);
          continue;
        }
      } else if (vali >= cutRegularEE2_ + energy * cutOverEEE2_ + energy * energy * cutOverE2EE2_) {
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
DEFINE_FWK_MODULE(HLTEgammaGenericQuadraticEtaFilter);
