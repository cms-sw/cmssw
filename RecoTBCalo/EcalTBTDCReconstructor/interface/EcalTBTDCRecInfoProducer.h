#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoProducer_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBTDCRecInfoProducer_HH

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoAlgo.h"

#include <vector>

class EcalTBTDCRecInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalTBTDCRecInfoProducer(const edm::ParameterSet& ps);
  ~EcalTBTDCRecInfoProducer() override;
  void produce(edm::Event& e, const edm::EventSetup& es) override;

private:
  std::string rawInfoProducer_;        // name of module/plugin/producer making digis
  std::string rawInfoCollection_;      // secondary name given to collection of digis
  std::string eventHeaderProducer_;    // name of module/plugin/producer making TBeventheader
  std::string eventHeaderCollection_;  // secondary name given to collection of TBeventheader
  std::string recInfoCollection_;      // secondary name to be given to collection of hits
  bool use2004OffsetConvention_;

  EcalTBTDCRecInfoAlgo* algo_;
};
#endif
