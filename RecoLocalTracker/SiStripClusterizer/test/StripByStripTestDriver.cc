
#include "RecoLocalTracker/SiStripClusterizer/test/StripByStripTestDriver.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/Event.h"

StripByStripTestDriver::StripByStripTestDriver(const edm::ParameterSet& conf)
    : inputTag(conf.getParameter<edm::InputTag>("DigiProducer")),
      hlt(conf.getParameter<bool>("HLT"))  //,
/*hltFactory(0)*/ {
  algorithm = StripClusterizerAlgorithmFactory::create(consumesCollector(), conf);

  produces<output_t>("");
}

StripByStripTestDriver::~StripByStripTestDriver() {
  //if(hltFactory) delete hltFactory;
}

void StripByStripTestDriver::produce(edm::Event& event, const edm::EventSetup& es) {
  auto output = std::make_unique<output_t>();
  output->reserve(10000, 4 * 10000);

  edm::Handle<edm::DetSetVector<SiStripDigi> > input;
  event.getByLabel(inputTag, input);

  algorithm->initialize(es);

  for (auto const& inputDetSet : *input) {
    output_t::TSFastFiller filler(*output, inputDetSet.detId());

    auto const& det = algorithm->stripByStripBegin(inputDetSet.detId());
    if (!det.valid())
      continue;
    StripClusterizerAlgorithm::State state(det);
    for (auto const& digi : inputDetSet)
      algorithm->stripByStripAdd(state, digi.strip(), digi.adc(), filler);
    algorithm->stripByStripEnd(state, filler);
  }

  edm::LogInfo("Output") << output->dataSize() << " clusters from " << output->size() << " modules";
  event.put(std::move(output));
}
