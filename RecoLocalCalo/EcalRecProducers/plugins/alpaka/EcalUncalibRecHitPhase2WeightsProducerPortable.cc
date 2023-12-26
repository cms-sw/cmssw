#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "DataFormats/EcalDigi/interface/alpaka/EcalDigiPhase2DeviceCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2HostCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitHostCollection.h"
#include "DataFormats/EcalRecHit/interface/alpaka/EcalUncalibratedRecHitDeviceCollection.h"

#include "EcalUncalibRecHitPhase2WeightsAlgoPortable.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EcalUncalibRecHitPhase2WeightsProducerPortable : public global::EDProducer<> {
  public:
    explicit EcalUncalibRecHitPhase2WeightsProducerPortable(edm::ParameterSet const &ps);
    ~EcalUncalibRecHitPhase2WeightsProducerPortable() override = default;
    static void fillDescriptions(edm::ConfigurationDescriptions &);

    void produce(edm::StreamID sid, device::Event &, device::EventSetup const &) const override;

  private:
    cms::alpakatools::host_buffer<double[]> weights_;
    cms::alpakatools::host_buffer<double[]> timeWeights_;

    using InputProduct = EcalDigiPhase2DeviceCollection;
    const device::EDGetToken<InputProduct> digisToken_;  //both tokens stored on the device
    using OutputProduct = EcalUncalibratedRecHitDeviceCollection;
    const device::EDPutToken<OutputProduct> uncalibratedRecHitsToken_;
  };

  // constructor with initialisation of elements
  EcalUncalibRecHitPhase2WeightsProducerPortable::EcalUncalibRecHitPhase2WeightsProducerPortable(
      const edm::ParameterSet &ps)
      : weights_{cms::alpakatools::make_host_buffer<double[]>(ecalPh2::sampleSize)},
        timeWeights_{cms::alpakatools::make_host_buffer<double[]>(ecalPh2::sampleSize)},
        digisToken_{consumes(ps.getParameter<edm::InputTag>("digisLabelEB"))},
        uncalibratedRecHitsToken_{produces(ps.getParameter<std::string>("uncalibratedRecHitsLabelEB"))} {
    //extracting the weights, for-loop to save them to the buffer even if the size is different than standard
    const auto weights = ps.getParameter<std::vector<double>>("weights");
    const auto timeWeights = ps.getParameter<std::vector<double>>("timeWeights");
    for (unsigned int i = 0; i < ecalPh2::sampleSize; ++i) {
      if (i < weights.size()) {
        weights_[i] = weights[i];
      } else {
        weights_[i] = 0;
      }
      if (i < timeWeights.size()) {
        timeWeights_[i] = timeWeights[i];
      } else {
        timeWeights_[i] = 0;
      }
    }
  }

  void EcalUncalibRecHitPhase2WeightsProducerPortable::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("uncalibratedRecHitsLabelEB", "EcalUncalibRecHitsEB");
    //The below weights values should be kept up to date with those on the CPU version of this module
    desc.add<std::vector<double>>("weights",
                                  {-0.121016,
                                   -0.119899,
                                   -0.120923,
                                   -0.0848959,
                                   0.261041,
                                   0.509881,
                                   0.373591,
                                   0.134899,
                                   -0.0233605,
                                   -0.0913195,
                                   -0.112452,
                                   -0.118596,
                                   -0.121737,
                                   -0.121737,
                                   -0.121737,
                                   -0.121737});
    desc.add<std::vector<double>>("timeWeights",
                                  {0.429452,
                                   0.442762,
                                   0.413327,
                                   0.858327,
                                   4.42324,
                                   2.04369,
                                   -3.42426,
                                   -4.16258,
                                   -2.36061,
                                   -0.725371,
                                   0.0727267,
                                   0.326005,
                                   0.402035,
                                   0.404287,
                                   0.434207,
                                   0.422775});

    desc.add<edm::InputTag>("digisLabelEB", edm::InputTag("simEcalUnsuppressedDigis", ""));

    descriptions.addWithDefaultLabel(desc);
  }

  void EcalUncalibRecHitPhase2WeightsProducerPortable::produce(edm::StreamID sid,
                                                               device::Event &event,
                                                               const device::EventSetup &setup) const {
    //get the device collection of digis
    auto const &digis = event.get(digisToken_);

    //get size of digis
    const uint32_t size = digis->metadata().size();

    //allocate output product on the device
    OutputProduct uncalibratedRecHits{static_cast<int32_t>(size), event.queue()};

    //do not run the algo if there are no digis
    if (size > 0) {
      //launch the asynchronous work
      ecal::weights::phase2Weights(digis, uncalibratedRecHits, weights_, timeWeights_, event.queue());
    }
    //put the output collection into the event
    event.emplace(uncalibratedRecHitsToken_, std::move(uncalibratedRecHits));
  }

}  //namespace ALPAKA_ACCELERATOR_NAMESPACE
DEFINE_FWK_ALPAKA_MODULE(EcalUncalibRecHitPhase2WeightsProducerPortable);
