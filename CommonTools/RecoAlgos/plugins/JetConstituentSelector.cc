/* \class PFJetSelector
 *
 * Selects jets with a configurable string-based cut,
 * and also writes out the constituents of the jet
 * into a separate collection.
 *
 * \author: Sal Rappoccio
 *
 *
 * for more details about the cut syntax, see the documentation
 * page below:
 *
 *   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
 *
 */

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

template <class T, typename C = std::vector<typename T::ConstituentTypeFwdPtr>>
class JetConstituentSelector : public edm::stream::EDProducer<> {
public:

  using JetsOutput = std::vector<T>;
  using ConstituentsOutput = C;

  JetConstituentSelector(edm::ParameterSet const& params) :
    srcToken_{consumes<edm::View<T>>(params.getParameter<edm::InputTag>("src"))},
    selector_{params.getParameter<std::string>("cut")}
  {
    produces<JetsOutput>();
    produces<ConstituentsOutput>("constituents");
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src")->setComment("InputTag used for retrieving jets in event.");
    desc.add<std::string>("cut")->setComment("Cut used by which to select jets.  For example:\n"
                                             "  \"pt > 100.0 && abs(rapidity()) < 2.4\".");

    // addDefault must be used here instead of add unless this function is specialized
    // for different sets of template parameter types. Each specialization would need
    // a different module label. Otherwise the generated cfi filenames will conflict
    // for the different plugins.
    descriptions.addDefault(desc);
  }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override
  {
    auto jets = std::make_unique<JetsOutput>();
    auto candsOut = std::make_unique<ConstituentsOutput>();

    edm::Handle<edm::View<T>> h_jets;
    iEvent.getByToken(srcToken_, h_jets);

    // Now set the Ptrs with the orphan handles.
    for (auto const& jet : *h_jets) {
      // Check the selection
      if (selector_(jet)) {
        // Add the jets that pass to the output collection
        jets->push_back(jet);
        for (unsigned int ida {}; ida < jet.numberOfDaughters(); ++ida) {
          candsOut->emplace_back(jet.daughterPtr(ida), jet.daughterPtr(ida));
        }
      }
    }

    iEvent.put(std::move(jets));
    iEvent.put(std::move(candsOut), "constituents");
  }

private:
  edm::EDGetTokenT<edm::View<T>> const srcToken_;
  StringCutObjectSelector<T> const selector_;
};

using PFJetConstituentSelector = JetConstituentSelector<reco::PFJet>;
using PatJetConstituentSelector = JetConstituentSelector<pat::Jet, std::vector<edm::FwdPtr<pat::PackedCandidate>>>;
using MiniAODJetConstituentSelector = JetConstituentSelector<reco::PFJet, std::vector<edm::FwdPtr<pat::PackedCandidate>>>;

DEFINE_FWK_MODULE(PFJetConstituentSelector);
DEFINE_FWK_MODULE(PatJetConstituentSelector);
DEFINE_FWK_MODULE(MiniAODJetConstituentSelector);
