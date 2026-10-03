#include <utility>

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/alpaka/BTLBaseRecHitDeviceCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/alpaka/BTLRecHitDeviceCollection.h"
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
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "BTLRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  using namespace ::btlrechit;

  class BTLRecHitSoAProducer : public stream::EDProducer<> {
  public:
    // constructor
    BTLRecHitSoAProducer(edm::ParameterSet const& config)
        : EDProducer<>(config),
          baserh_{consumes<::btlrechit::BTLBaseRecHitHostCollection>(config.getParameter<edm::InputTag>("baserh"))},
          rh_{produces()},
          invLightSpeedLYSO_(config.getParameter<double>("invLightSpeedLYSO")),
          c_LYSO_(1. / invLightSpeedLYSO_),
          thresholdToKeep_(config.getParameter<double>("thresholdToKeep")),
          calibration_(config.getParameter<double>("calibrationConstant")),
          npeSaturationCorr_(config.getParameter<std::vector<double>>("npeSaturationCorr")),
          npeToADC_(config.getParameter<std::vector<double>>("npeToADC")),
          npePerGeV_(config.getParameter<double>("npePerGeV")),
          timeCalibration_(config.getParameter<double>("timeCalibrationConstant")),
          tResParams_(config.getParameter<std::vector<double>>("tResParams")),
          twcParams_(config.getParameter<std::vector<double>>("twcParams")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("baserh");
      desc.add<double>("invLightSpeedLYSO");
      desc.add<double>("thresholdToKeep");
      desc.add<double>("calibrationConstant");
      desc.add<std::vector<double>>("npeSaturationCorr");
      desc.add<std::vector<double>>("npeToADC");
      desc.add<double>("npePerGeV");
      desc.add<double>("timeCalibrationConstant");
      desc.add<std::vector<double>>("tResParams");
      desc.add<std::vector<double>>("twcParams");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      // NB should be inserted a method to retrieve calibrations, now they are fixed to default values
      // Get the base from the Event.
      auto const& hostBrh = event.get(baserh_);  // SoA BaseRecHit stored in the event
      auto const N = hostBrh.const_view().metadata().size();

      // Copy input to device for GPU inference
      BTLBaseRecHitDeviceCollection deviceBrh(event.queue(), N);
      alpaka::memcpy(event.queue(), deviceBrh.buffer(), hostBrh.buffer());

      // Allocate a new SoA for the rechit.
      BTLRecHitDeviceCollection rh(event.queue(), N);

      // Apply the corrections and fill the new SoA. // these launch the kernel, and will run on gpu async
      std::array<double, 2> npeSaturationCorrArray_;
      std::array<double, 2> npeToADCArray_;
      std::array<double, 3> tResParamsArray_;
      std::array<double, 3> twcParamsArray_;
      std::copy_n(npeSaturationCorr_.begin(), 2, npeSaturationCorrArray_.begin());
      std::copy_n(npeToADC_.begin(), 2, npeToADCArray_.begin());
      std::copy_n(tResParams_.begin(), 3, tResParamsArray_.begin());
      std::copy_n(twcParams_.begin(), 3, twcParamsArray_.begin());
      BTLRecHitSoAProducerAlgo::fromBaseToReco(event.queue(),
                                               deviceBrh.view(),
                                               rh.view(),
                                               c_LYSO_,
                                               thresholdToKeep_,
                                               calibration_,
                                               npeSaturationCorrArray_,
                                               npeToADCArray_,
                                               npePerGeV_,
                                               timeCalibration_,
                                               tResParamsArray_,
                                               twcParamsArray_);

      // Move the SoA with the rh into the Event.
      event.emplace(rh_, std::move(rh));
    }

  private:
    const edm::EDGetTokenT<::btlrechit::BTLBaseRecHitHostCollection> baserh_;
    const device::EDPutToken<BTLRecHitDeviceCollection> rh_;
    const double invLightSpeedLYSO_;
    const double c_LYSO_;
    const double thresholdToKeep_;
    const double calibration_;
    const std::vector<double> npeSaturationCorr_;
    const std::vector<double> npeToADC_;
    const double npePerGeV_;
    const double timeCalibration_;
    const std::vector<double> tResParams_;
    const std::vector<double> twcParams_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(btlrechit::BTLRecHitSoAProducer);
