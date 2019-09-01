#ifndef EVENTFILTER_UTILITIES_PLUGINS_EVFFEDSELECTOR
#define EVENTFILTER_UTILITIES_PLUGINS_EVFFEDSELECTOR

#include <vector>

#include <FWCore/Framework/interface/global/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

namespace edm {
  class ConfigurationDescriptions;
}

namespace evf {

  class EvFFEDSelector : public edm::global::EDProducer<> {
  public:
    explicit EvFFEDSelector(edm::ParameterSet const &);
    ~EvFFEDSelector() override {}

    void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const final;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    edm::EDGetTokenT<FEDRawDataCollection> token_;
    std::vector<unsigned int> fedlist_;
  };

}  // namespace evf

#endif
