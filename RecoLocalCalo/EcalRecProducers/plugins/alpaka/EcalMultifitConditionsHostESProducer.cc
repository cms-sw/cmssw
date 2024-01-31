#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>
#include <cassert>
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitConditionsSoA.h"
#include "CondFormats/DataRecord/interface/EcalMultifitConditionsRcd.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EcalMultifitConditionsHostESProducer : public ESProducer {
  public:
    EcalMultifitConditionsHostESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      pedestalsToken_ = cc.consumes();
      gainRatiosToken_ = cc.consumes();
      pulseShapesToken_ = cc.consumes();
      pulseCovariancesToken_ = cc.consumes();
      samplesCorrelationToken_ = cc.consumes();
      timeBiasCorrectionsToken_ = cc.consumes();
      timeCalibConstantsToken_ = cc.consumes();
      sampleMaskToken_ = cc.consumes();
      timeOffsetConstantToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<EcalMultifitConditionsHost> produce(EcalMultifitConditionsRcd const& iRecord) {
      auto const& pedestalsData = iRecord.get(pedestalsToken_);
      auto const& gainRatiosData = iRecord.get(gainRatiosToken_);
      auto const& pulseShapesData = iRecord.get(pulseShapesToken_);
      auto const& pulseCovariancesData = iRecord.get(pulseCovariancesToken_);
      auto const& samplesCorrelationData = iRecord.get(samplesCorrelationToken_);
      auto const& timeBiasCorrectionsData = iRecord.get(timeBiasCorrectionsToken_);
      auto const& timeCalibConstantsData = iRecord.get(timeCalibConstantsToken_);
      auto const& sampleMaskData = iRecord.get(sampleMaskToken_);
      auto const& timeOffsetConstantData = iRecord.get(timeOffsetConstantToken_);

      size_t numberOfXtals = pedestalsData.size();

      auto product = std::make_unique<EcalMultifitConditionsHost>(numberOfXtals, cms::alpakatools::host());
      auto view = product->view();

      // Filling pedestals
      const auto barrelSize = pedestalsData.barrelItems().size();
      const auto endcapSize = pedestalsData.endcapItems().size();

      auto const& pedestalsEB = pedestalsData.barrelItems();
      auto const& pedestalsEE = pedestalsData.endcapItems();
      auto const& gainRatiosEB = gainRatiosData.barrelItems();
      auto const& gainRatiosEE = gainRatiosData.endcapItems();
      auto const& pulseShapesEB = pulseShapesData.barrelItems();
      auto const& pulseShapesEE = pulseShapesData.endcapItems();
      auto const& pulseCovariancesEB = pulseCovariancesData.barrelItems();
      auto const& pulseCovariancesEE = pulseCovariancesData.endcapItems();
      auto const& timeCalibConstantsEB = timeCalibConstantsData.barrelItems();
      auto const& timeCalibConstantsEE = timeCalibConstantsData.endcapItems();

      for (unsigned int i = 0; i < barrelSize; i++) {
        auto vi = view[i];

        vi.pedestals_mean_x12() = pedestalsEB[i].mean_x12;
        vi.pedestals_rms_x12() = pedestalsEB[i].rms_x12;
        vi.pedestals_mean_x6() = pedestalsEB[i].mean_x6;
        vi.pedestals_rms_x6() = pedestalsEB[i].rms_x6;
        vi.pedestals_mean_x1() = pedestalsEB[i].mean_x1;
        vi.pedestals_rms_x1() = pedestalsEB[i].rms_x1;

        vi.gain12Over6() = gainRatiosEB[i].gain12Over6();
        vi.gain6Over1() = gainRatiosEB[i].gain6Over1();

        vi.timeCalibConstants() = timeCalibConstantsEB[i];

        std::memcpy(vi.pulseShapes().data(), pulseShapesEB[i].pdfval, sizeof(float) * EcalPulseShape::TEMPLATESAMPLES);
        for (unsigned int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; j++) {
          for (unsigned int k = 0; k < EcalPulseShape::TEMPLATESAMPLES; k++) {
            vi.pulseCovariance()(j, k) = pulseCovariancesEB[i].val(j, k);
          }
        }
      }  // end Barrel loop
      for (unsigned int i = 0; i < endcapSize; i++) {
        auto vi = view[barrelSize + i];

        vi.pedestals_mean_x12() = pedestalsEE[i].mean_x12;
        vi.pedestals_rms_x12() = pedestalsEE[i].rms_x12;
        vi.pedestals_mean_x6() = pedestalsEE[i].mean_x6;
        vi.pedestals_rms_x6() = pedestalsEE[i].rms_x6;
        vi.pedestals_mean_x1() = pedestalsEE[i].mean_x1;
        vi.pedestals_rms_x1() = pedestalsEE[i].rms_x1;

        vi.gain12Over6() = gainRatiosEE[i].gain12Over6();
        vi.gain6Over1() = gainRatiosEE[i].gain6Over1();

        vi.timeCalibConstants() = timeCalibConstantsEE[i];

        std::memcpy(vi.pulseShapes().data(), pulseShapesEE[i].pdfval, sizeof(float) * EcalPulseShape::TEMPLATESAMPLES);

        for (unsigned int j = 0; j < EcalPulseShape::TEMPLATESAMPLES; j++) {
          for (unsigned int k = 0; k < EcalPulseShape::TEMPLATESAMPLES; k++) {
            vi.pulseCovariance()(j, k) = pulseCovariancesEE[i].val(j, k);
          }
        }
      }  // end Endcap loop

      // === Scalar data (not by xtal)
      //TimeBiasCorrection
      // Assert that there are not more parameters than the EcalMultiFitConditionsSoA expects
      assert(timeBiasCorrectionsData.EBTimeCorrAmplitudeBins.size() <= kMaxTimeBiasCorrectionBinsEB);
      assert(timeBiasCorrectionsData.EBTimeCorrShiftBins.size() <= kMaxTimeBiasCorrectionBinsEB);
      std::memcpy(view.timeBiasCorrections_amplitude_EB().data(),
                  timeBiasCorrectionsData.EBTimeCorrAmplitudeBins.data(),
                  sizeof(float) * kMaxTimeBiasCorrectionBinsEB);
      std::memcpy(view.timeBiasCorrections_shift_EB().data(),
                  timeBiasCorrectionsData.EBTimeCorrShiftBins.data(),
                  sizeof(float) * kMaxTimeBiasCorrectionBinsEB);

      // Assert that there are not more parameters than the EcalMultiFitConditionsSoA expects
      assert(timeBiasCorrectionsData.EETimeCorrAmplitudeBins.size() <= kMaxTimeBiasCorrectionBinsEE);
      assert(timeBiasCorrectionsData.EETimeCorrShiftBins.size() <= kMaxTimeBiasCorrectionBinsEE);
      std::memcpy(view.timeBiasCorrections_amplitude_EE().data(),
                  timeBiasCorrectionsData.EETimeCorrAmplitudeBins.data(),
                  sizeof(float) * kMaxTimeBiasCorrectionBinsEE);
      std::memcpy(view.timeBiasCorrections_shift_EE().data(),
                  timeBiasCorrectionsData.EETimeCorrShiftBins.data(),
                  sizeof(float) * kMaxTimeBiasCorrectionBinsEE);

      view.timeBiasCorrectionSizeEB() =
          std::min(timeBiasCorrectionsData.EBTimeCorrAmplitudeBins.size(), kMaxTimeBiasCorrectionBinsEB);
      view.timeBiasCorrectionSizeEE() =
          std::min(timeBiasCorrectionsData.EETimeCorrAmplitudeBins.size(), kMaxTimeBiasCorrectionBinsEE);

      // SampleCorrelation
      std::memcpy(view.sampleCorrelation_EB_G12().data(),
                  samplesCorrelationData.EBG12SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);
      std::memcpy(view.sampleCorrelation_EB_G6().data(),
                  samplesCorrelationData.EBG6SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);
      std::memcpy(view.sampleCorrelation_EB_G1().data(),
                  samplesCorrelationData.EBG1SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);

      std::memcpy(view.sampleCorrelation_EE_G12().data(),
                  samplesCorrelationData.EEG12SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);
      std::memcpy(view.sampleCorrelation_EE_G6().data(),
                  samplesCorrelationData.EBG6SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);
      std::memcpy(view.sampleCorrelation_EE_G1().data(),
                  samplesCorrelationData.EEG1SamplesCorrelation.data(),
                  sizeof(double) * ecalPh1::sampleSize);

      // Sample masks
      view.sampleMask_EB() = sampleMaskData.getEcalSampleMaskRecordEB();
      view.sampleMask_EE() = sampleMaskData.getEcalSampleMaskRecordEE();

      // Time offsets
      view.timeOffset_EB() = timeOffsetConstantData.getEBValue();
      view.timeOffset_EE() = timeOffsetConstantData.getEEValue();

      // number of barrel items as offset for hashed ID access to EE items of columns
      view.offsetEE() = barrelSize;

      return product;
    }

  private:
    edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
    edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainRatiosToken_;
    edm::ESGetToken<EcalPulseShapes, EcalPulseShapesRcd> pulseShapesToken_;
    edm::ESGetToken<EcalPulseCovariances, EcalPulseCovariancesRcd> pulseCovariancesToken_;
    edm::ESGetToken<EcalSamplesCorrelation, EcalSamplesCorrelationRcd> samplesCorrelationToken_;
    edm::ESGetToken<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd> timeBiasCorrectionsToken_;
    edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> timeCalibConstantsToken_;
    edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;
    edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> timeOffsetConstantToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(EcalMultifitConditionsHostESProducer);
