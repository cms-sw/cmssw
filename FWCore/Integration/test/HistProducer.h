#ifndef FWCore_Integration_HistProducer_h
#define FWCore_Integration_HistProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

namespace edmtest {
  //struct ThingWithHist {
  //	TH1F hist_;
  // };

  class HistProducer : public edm::global::EDProducer<> {
  public:
    explicit HistProducer(edm::ParameterSet const& ps);

    void produce(edm::StreamID, edm::Event& e, edm::EventSetup const& c) const override;

  private:
  };
}  // namespace edmtest
#endif
