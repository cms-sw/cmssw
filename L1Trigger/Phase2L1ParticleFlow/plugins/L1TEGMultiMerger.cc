#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include <iostream>

class L1TEGMultiMerger : public edm::global::EDProducer<> {
public:
  explicit L1TEGMultiMerger(const edm::ParameterSet&);
  ~L1TEGMultiMerger() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  template <class T>
  class InstanceMerger {
  public:
    InstanceMerger(L1TEGMultiMerger* prod, const edm::ParameterSet& conf)
        : instanceLabel_(conf.getParameter<std::string>("instance")) {
      for (const auto& producer_tag : conf.getParameter<std::vector<edm::InputTag>>("pfProducers")) {
        tokens_.push_back(
            prod->consumes<T>(edm::InputTag(producer_tag.label(), producer_tag.instance(), producer_tag.process())));
      }
      // FIXME: move this outside
      prod->produces<T>(instanceLabel_);
    }

    void produce(edm::Event& iEvent) const {
      edm::Handle<T> handle;
      auto out = std::make_unique<T>();
      for (const auto& token : tokens_) {
        iEvent.getByToken(token, handle);
        populate(out, handle);
      }
      iEvent.put(std::move(out), instanceLabel_);
    }

  private:
    template <class TT>
    void populate(std::unique_ptr<TT>& out, const edm::Handle<TT>& in) const {
      out->insert(out->end(), in->begin(), in->end());
    }

    void populate(std::unique_ptr<BXVector<l1t::EGamma>>& out, const edm::Handle<BXVector<l1t::EGamma>>& in) const {
      for (int bx = in->getFirstBX(); bx <= in->getLastBX(); bx++) {
        for (auto egee_itr = in->begin(bx); egee_itr != in->end(bx); egee_itr++) {
          out->push_back(bx, *egee_itr);
        }
      }
    }

    std::vector<edm::EDGetTokenT<T>> tokens_;
    std::string instanceLabel_;
  };

  std::vector<InstanceMerger<l1t::TkElectronCollection>> tkEleMerger;
  std::vector<InstanceMerger<l1t::TkEmCollection>> tkEmMerger;
  std::vector<InstanceMerger<BXVector<l1t::EGamma>>> tkEGMerger;
};

L1TEGMultiMerger::L1TEGMultiMerger(const edm::ParameterSet& conf) {
  for (const auto& config : conf.getParameter<std::vector<edm::ParameterSet>>("tkEgs")) {
    tkEGMerger.push_back(InstanceMerger<BXVector<l1t::EGamma>>(this, config));
  }
  for (const auto& config : conf.getParameter<std::vector<edm::ParameterSet>>("tkElectrons")) {
    tkEleMerger.push_back(InstanceMerger<l1t::TkElectronCollection>(this, config));
  }
  for (const auto& config : conf.getParameter<std::vector<edm::ParameterSet>>("tkEms")) {
    tkEmMerger.push_back(InstanceMerger<l1t::TkEmCollection>(this, config));
  }
}

L1TEGMultiMerger::~L1TEGMultiMerger() {}

void L1TEGMultiMerger::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  for (const auto& egMerger : tkEGMerger)
    egMerger.produce(iEvent);
  for (const auto& eleMerger : tkEleMerger)
    eleMerger.produce(iEvent);
  for (const auto& emMerger : tkEmMerger)
    emMerger.produce(iEvent);
}

void L1TEGMultiMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription psetDesc;
  psetDesc.add<std::string>("instance");
  psetDesc.add<std::vector<edm::InputTag>>("pfProducers");
  edm::ParameterSetDescription desc;
  desc.addVPSet("tkElectrons", psetDesc);
  desc.addVPSet("tkEms", psetDesc);
  desc.addVPSet("tkEgs", psetDesc);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TEGMultiMerger);
