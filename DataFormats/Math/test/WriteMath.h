#ifndef DataFormats_WriteMath_h
#define DataFormats_WriteMath_h
#include "FWCore/Framework/interface/global/EDProducer.h"

class WriteMath : public edm::global::EDProducer<> {
public:
  WriteMath(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const;
};

#endif
