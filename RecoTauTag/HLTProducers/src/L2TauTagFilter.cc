/*
 * \class L2TauTagFilter
 *
 * L2Tau identification using Convolutional NN.
 *
 * \author Valeria D'Amante, Università di Siena and INFN Pisa
 *         Konstantin Androsov, EPFL and ETHZ
 */

// user include files
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class L2TauTagFilter : public HLTFilter {
public:
  explicit L2TauTagFilter(const edm::ParameterSet& cfg)
      : HLTFilter(cfg),
        nExpected_(cfg.getParameter<int>("nExpected")),
        l1TauSrc_(cfg.getParameter<edm::InputTag>("L1TauSrc")),
        l1TauSrcToken_(consumes<trigger::TriggerFilterObjectWithRefs>(l1TauSrc_)),
        l2OutcomesToken_(consumes<std::vector<float>>(cfg.getParameter<edm::InputTag>("L2Outcomes"))),
        discrWP_(cfg.getParameter<double>("DiscrWP")),
        l1PtTh_(cfg.getParameter<double>("l1TauPtThreshold")) {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<int>("nExpected", 2)->setComment("number of expected taus per event");
    desc.add<edm::InputTag>("L1TauSrc", edm::InputTag(""))
        ->setComment("Which trigger should the L1 Taus collection pass");
    desc.add<edm::InputTag>("L2Outcomes", edm::InputTag(""))->setComment("L2 CNN outcomes");
    desc.add<double>("DiscrWP", 0.1227)->setComment("value of discriminator threshold");
    desc.add<double>("l1TauPtThreshold", 250)->setComment("value of L1Tau pass-through pt threshold");
    descriptions.addWithDefaultLabel(desc);
  }

  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& eventsetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override {
    if (saveTags())
      filterproduct.addCollectionTag(l1TauSrc_);

    int nTauPassed = 0;

    l1t::TauVectorRef l1Taus;
    auto const& l1TriggeredTaus = event.get(l1TauSrcToken_);
    l1TriggeredTaus.getObjects(trigger::TriggerL1Tau, l1Taus);

    auto const& L2Outcomes = event.get(l2OutcomesToken_);
    if (L2Outcomes.size() != l1Taus.size()) {
      throw cms::Exception("Inconsistent Data", "L2TauTagFilter::hltFilter") << "CNN output size != L1 taus size \n";
    }
    for (size_t l1_idx = 0; l1_idx < l1Taus.size(); l1_idx++) {
      if (L2Outcomes[l1_idx] >= discrWP_ || l1Taus[l1_idx]->pt() > l1PtTh_) {
        filterproduct.addObject(trigger::TriggerL1Tau, l1Taus[l1_idx]);
        nTauPassed++;
      }
    }

    return nTauPassed >= nExpected_;
  }

private:
  const int nExpected_;
  const edm::InputTag l1TauSrc_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> l1TauSrcToken_;
  const edm::EDGetTokenT<std::vector<float>> l2OutcomesToken_;
  const double discrWP_;
  const double l1PtTh_;
};

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2TauTagFilter);
