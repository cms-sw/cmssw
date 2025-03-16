#include <cassert>

#include <fmt/format.h>

#include "DataFormats/HeterogeneousTutorial/interface/JetsHostCollection.h"
#include "DataFormats/HeterogeneousTutorial/interface/TripletsHostCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace tutorial {

  namespace {

    // Build the InputTag for the backend
    edm::InputTag getBackendTag(edm::InputTag const& tag) {
      return edm::InputTag(tag.label(), "backend", tag.process());
    }

  }  // namespace

  class PFJetsSoAAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    PFJetsSoAAnalyzer(edm::ParameterSet const& config)
        : jets_{consumes(config.getParameter<edm::InputTag>("jets"))},
          soa_{consumes(config.getParameter<edm::InputTag>("soa"))},
          soa_backend_{consumes(getBackendTag(config.getParameter<edm::InputTag>("soa")))},
          ntuplets_{consumes(config.getParameter<edm::InputTag>("ntuplets"))},
          ntuplets_backend_{consumes(getBackendTag(config.getParameter<edm::InputTag>("ntuplets")))}  //
    {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("jets");
      desc.add<edm::InputTag>("soa");
      desc.add<edm::InputTag>("ntuplets");

      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
      edm::LogInfo msg("PFJetsSoAAnalyzer");

      auto const& jets = event.get(jets_);
      auto const& soa = event.get(soa_);
      auto soa_backend = static_cast<cms::alpakatools::Backend>(event.get(soa_backend_));
      msg << "SoA producer backend: " << cms::alpakatools::toString(soa_backend) << "\n";

      auto const& ntuplets = event.get(ntuplets_);
      auto ntuplets_backend = static_cast<cms::alpakatools::Backend>(event.get(ntuplets_backend_));
      msg << "N-tuplets producer backend: " << cms::alpakatools::toString(ntuplets_backend) << "\n";

      // TODO replace with an exception with a meaningful error message
      assert(static_cast<int>(jets.size()) == soa.view().metadata().size());

      int size = ntuplets.view().size();
      msg << "Found " << size << " N-tuplets:\n";

      for (int n = 0; n < size; ++n) {
        auto const& entry = ntuplets.view()[n];
        int i = entry.first();
        int j = entry.second();
        int k = entry.third();
        if (k == kEmpty) {
          msg << fmt::format("  jet pair: {}, {}\n", i, j);
          msg << "    " << jets[i] << " --> " << soa.view()[i] << '\n';
          msg << "    " << jets[j] << " --> " << soa.view()[j] << '\n';
        } else {
          msg << fmt::format("  jet triplet: {}, {}, {}\n", i, j, k);
          msg << "    " << jets[i] << " --> " << soa.view()[i] << '\n';
          msg << "    " << jets[j] << " --> " << soa.view()[j] << '\n';
          msg << "    " << jets[k] << " --> " << soa.view()[k] << '\n';
        }
      }
    }

  private:
    const edm::EDGetTokenT<reco::PFJetCollection> jets_;
    const edm::EDGetTokenT<JetsHostCollection> soa_;
    const edm::EDGetTokenT<unsigned short> soa_backend_;
    const edm::EDGetTokenT<TripletsHostCollection> ntuplets_;
    const edm::EDGetTokenT<unsigned short> ntuplets_backend_;
  };

}  // namespace tutorial

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(tutorial::PFJetsSoAAnalyzer);
