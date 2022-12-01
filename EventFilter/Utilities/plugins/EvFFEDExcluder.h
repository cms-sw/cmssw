#ifndef EventFilter_Utilities_EvFFEDExcluder_h
#define EventFilter_Utilities_EvFFEDExcluder_h

#include <vector>

#include "FWCore/Framework/interface/global/EDProducer.h"

class FEDRawDataCollection;

namespace evf {

  class EvFFEDExcluder : public edm::global::EDProducer<> {
  public:
    explicit EvFFEDExcluder(edm::ParameterSet const&);
    ~EvFFEDExcluder() override = default;

    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const final;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    edm::EDGetTokenT<FEDRawDataCollection> const rawDataToken_;
    std::vector<unsigned int> const fedIds_;
  };

}  // namespace evf

#endif  // EventFilter_Utilities_EvFFEDExcluder_h
