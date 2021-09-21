/*
 * \class L2TauTagFilter
 *
 * L2Tau identification using Convolutional NN.
 *
 * \author Valeria D'Amante, Università di Siena and INFN Pisa
 *         Konstantin Androsov, EPFL and ETHZ
 */

// system include files
#include <iostream>
// user include files
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace tau_hlt {

  class L2TauTagFilter : public HLTFilter {
  public:
    explicit L2TauTagFilter(const edm::ParameterSet& cfg)
        : HLTFilter(cfg),
          nExpected_(cfg.getParameter<int>("nExpected")),
          L1TauSrc_(cfg.getParameter<edm::InputTag>("L1TauSrc")),
          L1TauSrcToken_(consumes<trigger::TriggerFilterObjectWithRefs>(L1TauSrc_)),
          L2OutcomesToken_(consumes<std::vector<float>>(cfg.getParameter<edm::InputTag>("L2Outcomes"))),
          DiscrWP_(cfg.getParameter<double>("DiscrWP")) {}
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      makeHLTFilterDescription(desc);
      desc.add<int>("nExpected", 2)->setComment("number of expected taus per event");
      desc.add<edm::InputTag>("L1TauSrc", edm::InputTag(""))
          ->setComment("Which trigger should the L1 Taus collection pass");
      desc.add<edm::InputTag>("L2Outcomes", edm::InputTag(""))->setComment("L2 CNN outcomes");
      desc.add<double>("DiscrWP", 0.12267940863785043)->setComment("value of discriminator threshold");
      descriptions.addWithDefaultLabel(desc);
    }

    bool hltFilter(edm::Event& event,
                   const edm::EventSetup& eventsetup,
                   trigger::TriggerFilterObjectWithRefs& filterproduct) const override {
      if (saveTags())
        filterproduct.addCollectionTag(L1TauSrc_);

      int nTauPassed = 0;

      edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredTaus;
      event.getByToken(L1TauSrcToken_, l1TriggeredTaus);
      l1t::TauVectorRef L1Taus;
      l1TriggeredTaus->getObjects(trigger::TriggerL1Tau, L1Taus);

      edm::Handle<std::vector<float>> L2Outcomes_;
      event.getByToken(L2OutcomesToken_, L2Outcomes_);
      const auto L2Outcomes = *L2Outcomes_;
      if (L2Outcomes.size() != L1Taus.size()) {
        throw cms::Exception("Inconsistent Data", "L2TauTagFilter::hltFilter") << "CNN output size != L1 taus size \n";
      }
      for (size_t l1_idx = 0; l1_idx < L1Taus.size(); l1_idx++) {
        if (L2Outcomes[l1_idx] >= DiscrWP_) {
          filterproduct.addObject(nTauPassed, L1Taus[l1_idx]);
          nTauPassed++;
        }
      }

      return nTauPassed >= nExpected_;
    }

  private:
    const int nExpected_;
    const edm::InputTag L1TauSrc_;
    const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> L1TauSrcToken_;
    const edm::EDGetTokenT<std::vector<float>> L2OutcomesToken_;
    const double DiscrWP_;
  };

}  // namespace tau_hlt
using L2TauTagFilter = tau_hlt::L2TauTagFilter;
//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2TauTagFilter);
