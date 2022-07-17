#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo.h"

class PhotonConversionTrajectorySeedProducerFromQuadruplets : public edm::stream::EDProducer<> {
public:
  PhotonConversionTrajectorySeedProducerFromQuadruplets(const edm::ParameterSet&);
  ~PhotonConversionTrajectorySeedProducerFromQuadruplets() override {}
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  std::string _newSeedCandidates;
  std::unique_ptr<PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo> _theFinder;
};

PhotonConversionTrajectorySeedProducerFromQuadruplets::PhotonConversionTrajectorySeedProducerFromQuadruplets(
    const edm::ParameterSet& conf)
    : _newSeedCandidates(conf.getParameter<std::string>("newSeedCandidates")) {
  _theFinder = std::make_unique<PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo>(conf, consumesCollector());
  produces<TrajectorySeedCollection>(_newSeedCandidates);
}

void PhotonConversionTrajectorySeedProducerFromQuadruplets::produce(edm::Event& ev, const edm::EventSetup& es) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  try {
    _theFinder->analyze(ev, es);
    if (!_theFinder->getTrajectorySeedCollection()->empty())
      result->insert(result->end(),
                     _theFinder->getTrajectorySeedCollection()->begin(),
                     _theFinder->getTrajectorySeedCollection()->end());
  } catch (cms::Exception& er) {
    edm::LogError("SeedingConversion") << " Problem in the Single Leg Conversion Seed Producer " << er.what()
                                       << std::endl;
  } catch (std::exception& er) {
    edm::LogError("SeedingConversion") << " Problem in the Single Leg Conversion Seed Producer " << er.what()
                                       << std::endl;
  }

  edm::LogInfo("debugTrajSeedFromQuadruplets") << " TrajectorySeedCollection size " << result->size();
  ev.put(std::move(result), _newSeedCandidates);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonConversionTrajectorySeedProducerFromQuadruplets);
