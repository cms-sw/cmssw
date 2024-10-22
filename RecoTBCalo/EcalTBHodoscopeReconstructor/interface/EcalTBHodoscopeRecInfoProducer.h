#ifndef RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoProducer_HH
#define RecoTBCalo_EcalTBHodoscopeReconstructor_EcalTBHodoscopeRecInfoProducer_HH

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRecInfoAlgo.h"

#include <vector>

class EcalTBHodoscopeRecInfoProducer : public edm::global::EDProducer<> {
public:
  explicit EcalTBHodoscopeRecInfoProducer(const edm::ParameterSet& ps);

  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& es) const override;

private:
  edm::EDGetTokenT<EcalTBHodoscopeRawInfo> rawInfoProducerToken_;
  std::string rawInfoCollection_;  // secondary name given to collection of digis
  std::string recInfoCollection_;  // secondary name to be given to collection of hits

  int fitMethod_;
  EcalTBHodoscopeRecInfoAlgo algo_;
};
#endif
