#ifndef RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoProducer_HH
#define RecoTBCalo_EcalTBTDCReconstructor_EcalTBH2TDCRecInfoProducer_HH

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoAlgo.h"

#include <vector>

class EcalTBH2TDCRecInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalTBH2TDCRecInfoProducer(const edm::ParameterSet& ps);
  ~EcalTBH2TDCRecInfoProducer() override;
  void produce(edm::Event& e, const edm::EventSetup& es) override;

private:
  std::string rawInfoProducer_;        // name of module/plugin/producer making digis
  std::string rawInfoCollection_;      // secondary name given to collection of digis
  std::string triggerDataProducer_;    // name of module/plugin/producer making TBeventheader
  std::string triggerDataCollection_;  // secondary name given to collection of TBeventheader
  std::string recInfoCollection_;      // secondary name to be given to collection of hits

  EcalTBH2TDCRecInfoAlgo* algo_;
};
#endif
