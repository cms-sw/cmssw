#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "CLHEP/Random/RandFlat.h"

namespace edmtest {
  class RandomIntProducer : public edm::one::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    RandomIntProducer(edm::ParameterSet const& iPSet);

    void produce(edm::Event&, edm::EventSetup const&) final;

    void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) final;

  private:
    edm::EDPutTokenT<IntProduct> const evToken_;
    edm::EDPutTokenT<IntProduct> const lumiToken_;
  };
  RandomIntProducer::RandomIntProducer(edm::ParameterSet const&)
      : evToken_{produces<IntProduct>()},
        lumiToken_{produces<IntProduct, edm::Transition::BeginLuminosityBlock>("lumi")} {}

  void RandomIntProducer::produce(edm::Event& iEvent, edm::EventSetup const&) {
    edm::Service<edm::RandomNumberGenerator> gen;
    iEvent.emplace(evToken_, CLHEP::RandFlat::shootInt(&gen->getEngine(iEvent.streamID()), 10));
  }

  void RandomIntProducer::beginLuminosityBlockProduce(edm::LuminosityBlock& iLumi, edm::EventSetup const&) {
    edm::Service<edm::RandomNumberGenerator> gen;
    iLumi.emplace(lumiToken_, CLHEP::RandFlat::shootInt(&gen->getEngine(iLumi.index()), 10));
  }

}  // namespace edmtest

using namespace edmtest;
DEFINE_FWK_MODULE(RandomIntProducer);
