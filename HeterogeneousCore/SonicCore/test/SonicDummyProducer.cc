#include "DummyClient.h"
#include "SonicCMS/Core/interface/SonicEDProducer.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>

namespace edmtest {
	template <typename Client>
	class SonicDummyProducer : public SonicEDProducer<Client> {
		public:
			//needed because base class has dependent scope
			using typename SonicEDProducer<Client>::Input;
			using typename SonicEDProducer<Client>::Output;
			explicit SonicDummyProducer(edm::ParameterSet const& cfg) :
				SonicEDProducer<Client>(cfg),
				input_(cfg.getParameter<int>("input"))
			{
				this->template produces<IntProduct>();
			}

			void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
				iInput = input_;
			}

			void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
				iEvent.put(std::make_unique<IntProduct>(iOutput));
			}

			//to ensure distinct cfi names - specialized below
			static std::string getCfiName();
			static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
				edm::ParameterSetDescription desc;
				Client::fillPSetDescription(desc);
				desc.add<int>("input");
				descriptions.add(getCfiName(),desc);
			}

		private:
			//members
			int input_;
	};

	typedef SonicDummyProducer<DummyClientSync> SonicDummyProducerSync;
	typedef SonicDummyProducer<DummyClientPseudoAsync> SonicDummyProducerPseudoAsync;
	typedef SonicDummyProducer<DummyClientAsync> SonicDummyProducerAsync;

	template<> std::string SonicDummyProducerSync::getCfiName() { return "SonicDummyProducerSync"; }
	template<> std::string SonicDummyProducerPseudoAsync::getCfiName() { return "SonicDummyProducerPseudoAsync"; }
	template<> std::string SonicDummyProducerAsync::getCfiName() { return "SonicDummyProducerAsync"; }
}

using edmtest::SonicDummyProducerSync;
DEFINE_FWK_MODULE(SonicDummyProducerSync);
using edmtest::SonicDummyProducerPseudoAsync;
DEFINE_FWK_MODULE(SonicDummyProducerPseudoAsync);
using edmtest::SonicDummyProducerAsync;
DEFINE_FWK_MODULE(SonicDummyProducerAsync);

