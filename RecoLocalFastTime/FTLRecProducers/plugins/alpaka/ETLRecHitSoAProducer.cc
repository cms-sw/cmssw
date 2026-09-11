#include <utility>

#include "CommonTools/Utils/interface/FormulaEvaluator.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/alpaka/ETLBaseRecHitDeviceCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/alpaka/ETLRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "ETLRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using namespace ::etlrechit;

  class ETLRecHitSoAProducer : public stream::EDProducer<> {
  public:
    // constructor
    ETLRecHitSoAProducer(edm::ParameterSet const& config)
        : EDProducer<>(config),
          baserh_{consumes<::etlrechit::ETLBaseRecHitHostCollection>(config.getParameter<edm::InputTag>("baserh"))},
          rh_{produces()},
          thresholdToKeep_(config.getParameter<double>("thresholdToKeep")),
          calibration_(config.getParameter<double>("calibrationConstant")),
          timeResInNs_(config.getParameter<double>("timeResInNs")),
          timeCorr_p0_(config.getParameter<double>("timeCorr_p0")),
          timeCorr_p1_(config.getParameter<double>("timeCorr_p1")),
          timeCorr_p2_(config.getParameter<double>("timeCorr_p2")),
          timeCorr_p3_(config.getParameter<double>("timeCorr_p3")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("baserh");
      desc.add<double>("thresholdToKeep");
      desc.add<double>("calibrationConstant");
      desc.add<double>("timeResInNs");
      desc.add<double>("timeCorr_p0");
      desc.add<double>("timeCorr_p1");
      desc.add<double>("timeCorr_p2");
      desc.add<double>("timeCorr_p3");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      // NB should be inserted a method to retrieve calibrations, now they are fixed to default values
      // Get the base from the Event.
      auto const& hostBrh = event.get(baserh_);  // SoA BaseRecHit stored in the event
      auto const N = hostBrh.const_view().metadata().size();

      // Copy input to device for GPU inference
      ETLBaseRecHitDeviceCollection deviceBrh(event.queue(), N);
      alpaka::memcpy(event.queue(), deviceBrh.buffer(), hostBrh.buffer());

      // Allocate a new SoA for the rechit.
      ETLRecHitDeviceCollection rh(event.queue(), N);

      // Apply the corrections and fill the new SoA. // these launch the kernel, and will run on gpu async
      ETLRecHitSoAProducerAlgo::fromBaseToReco(event.queue(),
                                               deviceBrh.view(),
                                               rh.view(),
                                               thresholdToKeep_,
                                               calibration_,
                                               timeResInNs_,
                                               timeCorr_p0_,
                                               timeCorr_p2_,
                                               timeCorr_p1_,
                                               timeCorr_p3_);

      // Move the SoA with the rh into the Event.
      event.emplace(rh_, std::move(rh));
    }

  private:
    const edm::EDGetTokenT<::etlrechit::ETLBaseRecHitHostCollection> baserh_;
    const device::EDPutToken<ETLRecHitDeviceCollection> rh_;
    //edm::ESGetToken<MTDTimeCalib, MTDTimeCalibRecord> tcToken_;
    const double thresholdToKeep_;
    const double calibration_;
    const double timeResInNs_;
    const double timeCorr_p0_;
    const double timeCorr_p1_;
    const double timeCorr_p2_;
    const double timeCorr_p3_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(etlrechit::ETLRecHitSoAProducer);
