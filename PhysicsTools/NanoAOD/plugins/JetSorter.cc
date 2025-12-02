#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include <algorithm>
#include <string>
#include <memory>

class JetSorter : public edm::global::EDProducer<> {
public:
  explicit JetSorter(const edm::ParameterSet &iConfig)
      : srcToken_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("src"))),
        userFloatSorter_(iConfig.getParameter<std::string>("userFloatSorter")),
        descending_(iConfig.getParameter<bool>("descending")) {
    produces<std::vector<pat::Jet>>();
  }

  ~JetSorter() override = default;

  void produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &) const override {
    edm::Handle<edm::View<pat::Jet>> hSrc;
    iEvent.getByToken(srcToken_, hSrc);

    auto out = std::make_unique<std::vector<pat::Jet>>();
    out->reserve(hSrc->size());

    for (auto const &jIn : *hSrc) {
      out->push_back(jIn);
    }

    auto key = [this](pat::Jet const &j) -> float {
      if (j.hasUserFloat(userFloatSorter_)) {
        return j.userFloat(userFloatSorter_);
      }
      return static_cast<float>(j.pt());
    };

    std::stable_sort(out->begin(), out->end(), [this, &key](pat::Jet const &a, pat::Jet const &b) {
      float pa = key(a);
      float pb = key(b);
      return descending_ ? (pa > pb) : (pa < pb);
    });

    iEvent.put(std::move(out));
  }

private:
  edm::EDGetTokenT<edm::View<pat::Jet>> srcToken_;
  std::string userFloatSorter_;
  bool descending_;
};

DEFINE_FWK_MODULE(JetSorter);
