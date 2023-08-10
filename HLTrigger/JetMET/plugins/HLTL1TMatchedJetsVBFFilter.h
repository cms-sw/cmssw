#ifndef HLTrigger_JetMET_HLTL1TMatchedJetsVBFFilter_h
#define HLTrigger_JetMET_HLTL1TMatchedJetsVBFFilter_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <vector>
#include <utility>
#include <string>

template <typename T>
class HLTL1TMatchedJetsVBFFilter : public HLTFilter {
public:
  explicit HLTL1TMatchedJetsVBFFilter(const edm::ParameterSet&);
  ~HLTL1TMatchedJetsVBFFilter() override = default;

  bool hltFilter(edm::Event& iEvent,
                 const edm::EventSetup& iSetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  auto logTrace() const {
    auto const& moduleType = moduleDescription().moduleName();
    auto const& moduleLabel = moduleDescription().moduleLabel();
    return LogTrace(moduleType) << "[" << moduleType << "] (" << moduleLabel << ") ";
  }

  enum Algorithm { VBF = 0, VBFPlus2CentralJets = 1 };

  Algorithm getAlgorithmFromString(std::string const& str) const;
  void fillJetIndices(std::vector<unsigned int>& outputJetIndices,
                      std::vector<T> const& jets,
                      std::vector<unsigned int> const& jetIndices,
                      double const pt1,
                      double const pt3,
                      double const mjj) const;

  // input HLT jets
  edm::InputTag const jetTag_;
  edm::EDGetTokenT<std::vector<T>> const jetToken_;

  // matching with L1T jets by delta-R distance
  bool const matchJetsByDeltaR_;
  double const maxJetDeltaR_, maxJetDeltaR2_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> const l1tJetRefsToken_;

  // selection of HLT jets compatible with the VBF topology (see fillJetIndices)
  Algorithm const algorithm_;
  double const minPt1_;
  double const minPt2_;
  double const minPt3_;
  double const minInvMass_;

  // min/max number of selected jets, and triggerType of output triggerRefs
  int const minNJets_;
  int const maxNJets_;
  int const triggerType_;
};

template <typename T>
void HLTL1TMatchedJetsVBFFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("src", edm::InputTag("hltJets"))->setComment("InputTag of input HLT jets");

  desc.add<bool>("matchJetsByDeltaR", true)
      ->setComment("Enable delta-R matching between HLT jets ('src') and L1T jets ('l1tJetRefs')");
  desc.add<double>("maxJetDeltaR", 0.5)
      ->setComment(
          "Maximum delta-R distance to match HLT jets ('src') and L1T jets ('l1tJetRefs'), "
          "used only if 'matchJetsByDeltaR == true'.");
  desc.add<edm::InputTag>("l1tJetRefs", edm::InputTag("hltL1sJetSeed"))
      ->setComment("InputTag of references to L1T jets");

  desc.add<std::string>("algorithm", "VBF")
      ->setComment("Keyword to choose the algorithm used to select jets compatible with the VBF topology");
  desc.add<double>("minPt1", 110.)
      ->setComment("Minimum pT threshold 'pt1' in algorithm to select VBF jets (must respect 'pt2 <= pt3 <= pt1')");
  desc.add<double>("minPt2", 35.)->setComment("Minimum pT of all selected VBF jets (must respect 'pt2 <= pt3 <= pt1')");
  desc.add<double>("minPt3", 110.)
      ->setComment("Minimum pT threshold 'pt3' in algorithm to select VBF jets (must respect 'pt2 <= pt3 <= pt1')");
  desc.add<double>("minInvMass", 650.)->setComment("Minimum jet invariant mass");

  desc.add<int>("minNJets", 0)->setComment("Minimum number of output jets to accept the event (ignored if negative)");
  desc.add<int>("maxNJets", -1)->setComment("Maximum number of output jets to accept the event (ignored if negative)");

  desc.add<int>("triggerType", trigger::TriggerJet);

  descriptions.setComment(
      "This HLTFilter selects events with HLT jets compatible with the VBF topology. "
      "These HLT jets can be required to be matched to L1T jets by delta-R distance.");
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
HLTL1TMatchedJetsVBFFilter<T>::HLTL1TMatchedJetsVBFFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      jetTag_(iConfig.getParameter<edm::InputTag>("src")),
      jetToken_(consumes(jetTag_)),
      matchJetsByDeltaR_(iConfig.getParameter<bool>("matchJetsByDeltaR")),
      maxJetDeltaR_(iConfig.getParameter<double>("maxJetDeltaR")),
      maxJetDeltaR2_(maxJetDeltaR_ * maxJetDeltaR_),
      l1tJetRefsToken_(matchJetsByDeltaR_ ? consumes<trigger::TriggerFilterObjectWithRefs>(
                                                iConfig.getParameter<edm::InputTag>("l1tJetRefs"))
                                          : edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>()),
      algorithm_(getAlgorithmFromString(iConfig.getParameter<std::string>("algorithm"))),
      minPt1_(iConfig.getParameter<double>("minPt1")),
      minPt2_(iConfig.getParameter<double>("minPt2")),
      minPt3_(iConfig.getParameter<double>("minPt3")),
      minInvMass_(iConfig.getParameter<double>("minInvMass")),
      minNJets_(iConfig.getParameter<int>("minNJets")),
      maxNJets_(iConfig.getParameter<int>("maxNJets")),
      triggerType_(iConfig.getParameter<int>("triggerType")) {
  // delta-R matching
  if (matchJetsByDeltaR_ and maxJetDeltaR_ < 0) {
    throw cms::Exception("InvalidConfigurationParameter")
        << "invalid value for parameter \"maxJetDeltaR\" (must be > 0): " << maxJetDeltaR_;
  }

  // values of minPt{1,2,3} thresholds
  if (minPt2_ > minPt1_) {
    throw cms::Exception("InvalidConfigurationParameter")
        << "minPt1 (" << minPt1_ << ") must not be smaller than minPt2 (" << minPt2_ << ")";
  } else if (minPt3_ > minPt1_) {
    throw cms::Exception("InvalidConfigurationParameter")
        << "minPt1 (" << minPt1_ << ") must not be smaller than minPt3 (" << minPt3_ << ")";
  } else if (minPt2_ > minPt3_) {
    throw cms::Exception("InvalidConfigurationParameter")
        << "minPt3 (" << minPt3_ << ") must not be smaller than minPt2 (" << minPt2_ << ")";
  }
}

template <typename T>
typename HLTL1TMatchedJetsVBFFilter<T>::Algorithm HLTL1TMatchedJetsVBFFilter<T>::getAlgorithmFromString(
    std::string const& str) const {
  if (str == "VBF") {
    return Algorithm::VBF;
  } else if (str == "VBFPlus2CentralJets") {
    return Algorithm::VBFPlus2CentralJets;
  }

  throw cms::Exception("HLTL1TMatchedJetsVBFFilterInvalidAlgorithmName")
      << "invalid value for argument of getAlgorithmFromString method: " << str
      << " (valid values are \"VBF\" and \"VBFPlus2CentralJets\")";
}

template <typename T>
void HLTL1TMatchedJetsVBFFilter<T>::fillJetIndices(std::vector<unsigned int>& outputJetIndices,
                                                   std::vector<T> const& jets,
                                                   std::vector<unsigned int> const& jetIndices,
                                                   double const pt1,
                                                   double const pt3,
                                                   double const mjj) const {
  // reset output jet indices
  outputJetIndices.clear();

  // find dijet pair with highest Mjj
  int i1 = -1;
  int i2 = -1;
  double m2jj_max = -1;

  for (unsigned int i = 0; i < jetIndices.size() - 1; i++) {
    auto const& jet1 = jets[jetIndices[i]];

    for (unsigned int j = i + 1; j < jetIndices.size(); j++) {
      auto const& jet2 = jets[jetIndices[j]];

      double const m2jj_tmp = (jet1.p4() + jet2.p4()).M2();

      if (m2jj_tmp > m2jj_max) {
        m2jj_max = m2jj_tmp;
        i1 = jetIndices[i];
        i2 = jetIndices[j];
      }
    }
  }

  // only proceed if highest-Mjj dijet pair was found
  if (i1 >= 0 and i2 >= 0) {
    logTrace() << "dijet pair with highest invariant mass: indices={" << i1 << "," << i2 << "} mjj^2=" << m2jj_max;

    unsigned int const j1 = i1;
    unsigned int const j2 = i2;

    auto const& jet1 = jets[j1];
    double const m2jj = (mjj >= 0. ? mjj * mjj : -1.);

    // algorithm: "VBF"
    if (algorithm_ == Algorithm::VBF) {
      if (m2jj_max > m2jj) {
        if (jet1.pt() >= pt1) {
          outputJetIndices = {j1, j2};
        } else if (jet1.pt() < pt3 and jets[jetIndices[0]].pt() > pt3) {
          outputJetIndices = {j1, j2, jetIndices[0]};
        }
      }
    } else if (algorithm_ == Algorithm::VBFPlus2CentralJets) {
      if (jetIndices.size() > 3 and m2jj_max > m2jj) {
        // indices of additional jets
        std::vector<unsigned int> idx_jets;
        idx_jets.reserve(jetIndices.size() - 2);

        for (auto const idx : jetIndices) {
          if (idx == j1 or idx == j2)
            continue;
          idx_jets.emplace_back(idx);
        }

        if (jet1.pt() >= pt3) {
          outputJetIndices.emplace_back(j1);
          outputJetIndices.emplace_back(j2);

          for (auto const idx : idx_jets) {
            if (outputJetIndices.size() > 3)
              break;
            outputJetIndices.emplace_back(idx);
          }
        } else if (idx_jets.size() >= 3) {
          outputJetIndices.emplace_back(j1);
          outputJetIndices.emplace_back(j2);
          outputJetIndices.emplace_back(idx_jets[0]);
          outputJetIndices.emplace_back(idx_jets[1]);
          outputJetIndices.emplace_back(idx_jets[2]);

          if (idx_jets.size() > 3) {
            outputJetIndices.emplace_back(idx_jets[3]);
          }
        }
      }
    }
  }
}

template <typename T>
bool HLTL1TMatchedJetsVBFFilter<T>::hltFilter(edm::Event& iEvent,
                                              const edm::EventSetup&,
                                              trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(jetTag_);

  // handle to input HLT jets
  auto const& jetsHandle = iEvent.getHandle(jetToken_);

  // indices of input HLT jets satisfying delta-R matching requirements
  std::vector<unsigned int> jetIndices;

  if (matchJetsByDeltaR_) {
    // refs to L1T jets used for delta-R matching
    l1t::JetVectorRef l1tJetRefs;
    iEvent.get(l1tJetRefsToken_).getObjects(trigger::TriggerL1Jet, l1tJetRefs);

    jetIndices.reserve(jetsHandle->size());
    for (unsigned int iJet = 0; iJet < jetsHandle->size(); ++iJet) {
      auto const& jet = (*jetsHandle)[iJet];

      if (jet.pt() <= minPt2_)
        continue;

      for (unsigned int l1tJetIdx = 0; l1tJetIdx < l1tJetRefs.size(); ++l1tJetIdx) {
        if (reco::deltaR2(jet.p4(), l1tJetRefs[l1tJetIdx]->p4()) < maxJetDeltaR2_) {
          logTrace() << "input jet: index=" << iJet << " pt=" << jet.pt() << " eta=" << jet.eta()
                     << " phi=" << jet.phi() << " mass=" << jet.mass()
                     << " (matched to L1T jet with pt=" << l1tJetRefs[l1tJetIdx]->pt()
                     << " eta=" << l1tJetRefs[l1tJetIdx]->eta() << " phi=" << l1tJetRefs[l1tJetIdx]->phi()
                     << " mass=" << l1tJetRefs[l1tJetIdx]->mass() << ")";

          jetIndices.emplace_back(iJet);
          break;
        }
      }
    }
  } else {
    jetIndices.reserve(jetsHandle->size());
    for (unsigned int iJet = 0; iJet < jetsHandle->size(); ++iJet) {
      auto const& jet = (*jetsHandle)[iJet];

      if (jet.pt() <= minPt2_)
        continue;

      logTrace() << "input jet: index=" << iJet << " pt=" << jet.pt() << " eta=" << jet.eta() << " phi=" << jet.phi()
                 << " mass=" << jet.mass();

      jetIndices.emplace_back(iJet);
    }
  }

  // order jet indices by jet pT (highest pT first)
  std::sort(jetIndices.begin(), jetIndices.end(), [&jetsHandle](unsigned int const i1, unsigned int const i2) {
    return (*jetsHandle)[i1].pt() > (*jetsHandle)[i2].pt();
  });

  // indices of jets identified as compatible with the VBF topology
  std::vector<unsigned int> outputJetIndices;
  fillJetIndices(outputJetIndices, *jetsHandle, jetIndices, minPt1_, minPt3_, minInvMass_);

  // add selected jets to trigger::TriggerFilterObjectWithRefs
  for (auto const idx : outputJetIndices) {
    filterproduct.addObject(triggerType_, edm::Ref<std::vector<T>>(jetsHandle, idx));

    logTrace() << "output jet: index=" << idx << " pt=" << (*jetsHandle)[idx].pt()
               << " eta=" << (*jetsHandle)[idx].eta() << " phi=" << (*jetsHandle)[idx].phi()
               << " mass=" << (*jetsHandle)[idx].mass();
  }

  // accept event only if minNJets and maxNJets requirements are satisfied
  auto const ret = ((minNJets_ < 0 or outputJetIndices.size() >= (unsigned int)minNJets_) and
                    (maxNJets_ < 0 or outputJetIndices.size() <= (unsigned int)maxNJets_));

  logTrace() << "filter decision = " << ret << " (trigger objects = " << outputJetIndices.size() << ")";

  return ret;
}

#endif
