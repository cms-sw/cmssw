// system includes
#include <memory>

// user includes
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"

class StripByStripTestDriver : public edm::stream::EDProducer<> {
  typedef edmNew::DetSetVector<SiStripCluster> output_t;

public:
  StripByStripTestDriver(const edm::ParameterSet&);
  ~StripByStripTestDriver();

private:
  void produce(edm::Event&, const edm::EventSetup&);

  const edm::InputTag inputTag;
  const edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>> siStripDigisToken;
  const bool hlt;

  std::unique_ptr<StripClusterizerAlgorithm> algorithm;
};

StripByStripTestDriver::StripByStripTestDriver(const edm::ParameterSet& conf)
    : inputTag(conf.getParameter<edm::InputTag>("DigiProducer")),
      siStripDigisToken(consumes<edm::DetSetVector<SiStripDigi>>(inputTag)),
      hlt(conf.getParameter<bool>("HLT")) {
  algorithm = StripClusterizerAlgorithmFactory::create(consumesCollector(), conf);

  produces<output_t>("");
}

StripByStripTestDriver::~StripByStripTestDriver() = default;

void StripByStripTestDriver::produce(edm::Event& event, const edm::EventSetup& es) {
  auto output = std::make_unique<output_t>();
  output->reserve(10000, 4 * 10000);

  edm::Handle<edm::DetSetVector<SiStripDigi>> input;
  event.getByToken(siStripDigisToken, input);

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StripByStripTestDriver);
