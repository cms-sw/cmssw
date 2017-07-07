/*
 * Produce an int in the event that specifies an event type.
 *
 * Takes one int parameter: flag, and puts it in the event
 *
 * Used to identify different samples in MVA training.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RecoTauEventFlagProducer : public edm::EDProducer {
  public:
    RecoTauEventFlagProducer(const edm::ParameterSet &pset) {
      flag_ = pset.getParameter<int>("flag");
      produces<int>();
    }
    ~RecoTauEventFlagProducer() override {}
    void produce(edm::Event& evt, const edm::EventSetup &es) override {
      evt.put(std::make_unique<int>(flag_));
    }
  private:
    int flag_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauEventFlagProducer);
