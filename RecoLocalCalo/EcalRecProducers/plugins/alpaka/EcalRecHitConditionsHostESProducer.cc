#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

#include "CondFormats/EcalObjects/interface/alpaka/EcalRecHitConditionsDevice.h"
#include "CondFormats/EcalObjects/interface/EcalRecHitConditionsSoA.h"
#include "CondFormats/DataRecord/interface/EcalRecHitConditionsRcd.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EcalRecHitConditionsHostESProducer : public ESProducer {
  public:
    EcalRecHitConditionsHostESProducer(edm::ParameterSet const& iConfig)
        : ESProducer(iConfig), isPhase2_{iConfig.getParameter<bool>("isPhase2")} {
      auto cc = setWhatProduced(this);
      adcToGeVConstantToken_ = cc.consumes();
      intercalibConstantsToken_ = cc.consumes();
      channelStatusToken_ = cc.consumes();
      laserAPDPNRatiosToken_ = cc.consumes();
      laserAPDPNRatiosRefToken_ = cc.consumes();
      laserAlphasToken_ = cc.consumes();
      linearCorrectionsToken_ = cc.consumes();
      timeCalibConstantsToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("timeCalibTag"));
      timeOffsetConstantToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("timeOffsetTag"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("timeCalibTag", edm::ESInputTag());
      desc.add<edm::ESInputTag>("timeOffsetTag", edm::ESInputTag());
      desc.add<bool>("isPhase2", false);
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<EcalRecHitConditionsHost> produce(EcalRecHitConditionsRcd const& iRecord) {
      auto const& adcToGeVConstantData = iRecord.get(adcToGeVConstantToken_);
      auto const& intercalibConstantsData = iRecord.get(intercalibConstantsToken_);
      auto const& channelStatusData = iRecord.get(channelStatusToken_);
      auto const& laserAPDPNRatiosData = iRecord.get(laserAPDPNRatiosToken_);
      auto const& laserAPDPNRatiosRefData = iRecord.get(laserAPDPNRatiosRefToken_);
      auto const& laserAlphasData = iRecord.get(laserAlphasToken_);
      auto const& linearCorrectionsData = iRecord.get(linearCorrectionsToken_);
      auto const& timeCalibConstantsData = iRecord.get(timeCalibConstantsToken_);
      auto const& timeOffsetConstantData = iRecord.get(timeOffsetConstantToken_);

      const auto barrelSize = channelStatusData.barrelItems().size();
      auto numberOfXtals = barrelSize;
      if (!isPhase2_) {
        numberOfXtals += channelStatusData.endcapItems().size();
      }

      auto product = std::make_unique<EcalRecHitConditionsHost>(numberOfXtals, cms::alpakatools::host());
      auto view = product->view();

      // Filling crystal level conditions
      auto const& intercalibConstantsEB = intercalibConstantsData.barrelItems();
      auto const& channelStatusEB = channelStatusData.barrelItems();
      auto const& laserAPDPNRatiosLaserEB = laserAPDPNRatiosData.getLaserMap().barrelItems();
      auto const& laserAPDPNRatiosTime = laserAPDPNRatiosData.getTimeMap();
      auto const& laserAPDPNRatiosRefEB = laserAPDPNRatiosRefData.barrelItems();
      auto const& laserAlphasEB = laserAlphasData.barrelItems();
      auto const& linearCorrectionsEB = linearCorrectionsData.getValueMap().barrelItems();
      auto const& linearCorrectionsTime = linearCorrectionsData.getTimeMap();
      auto const& timeCalibConstantsEB = timeCalibConstantsData.barrelItems();

      // barrel conditions
      for (unsigned int i = 0; i < barrelSize; ++i) {
        auto vi = view[i];

        vi.intercalibConstants() = intercalibConstantsEB[i];
        vi.channelStatus() = channelStatusEB[i].getEncodedStatusCode();

        vi.laserAPDPNRatios_p1() = laserAPDPNRatiosLaserEB[i].p1;
        vi.laserAPDPNRatios_p2() = laserAPDPNRatiosLaserEB[i].p2;
        vi.laserAPDPNRatios_p3() = laserAPDPNRatiosLaserEB[i].p3;

        vi.laserAPDPNref() = laserAPDPNRatiosRefEB[i];
        vi.laserAlpha() = laserAlphasEB[i];

        vi.linearCorrections_p1() = linearCorrectionsEB[i].p1;
        vi.linearCorrections_p2() = linearCorrectionsEB[i].p2;
        vi.linearCorrections_p3() = linearCorrectionsEB[i].p3;

        vi.timeCalibConstants() = timeCalibConstantsEB[i];
      }  // end Barrel loop

      // time maps
      for (unsigned int i = 0; i < laserAPDPNRatiosData.getTimeMap().size(); ++i) {
        auto vi = view[i];
        vi.laserAPDPNRatios_t1() = laserAPDPNRatiosTime[i].t1.value();
        vi.laserAPDPNRatios_t2() = laserAPDPNRatiosTime[i].t2.value();
        vi.laserAPDPNRatios_t3() = laserAPDPNRatiosTime[i].t3.value();
      }

      for (unsigned int i = 0; i < linearCorrectionsData.getTimeMap().size(); ++i) {
        auto vi = view[i];
        vi.linearCorrections_t1() = linearCorrectionsTime[i].t1.value();
        vi.linearCorrections_t2() = linearCorrectionsTime[i].t2.value();
        vi.linearCorrections_t3() = linearCorrectionsTime[i].t3.value();
      }

      // scalar data
      // ADC to GeV constants
      view.adcToGeVConstantEB() = adcToGeVConstantData.getEBValue();

      // time offset constants
      view.timeOffsetConstantEB() = timeOffsetConstantData.getEBValue();

      // endcap conditions
      if (!isPhase2_) {
        auto const& intercalibConstantsEE = intercalibConstantsData.endcapItems();
        auto const& channelStatusEE = channelStatusData.endcapItems();
        auto const& laserAPDPNRatiosLaserEE = laserAPDPNRatiosData.getLaserMap().endcapItems();
        auto const& laserAPDPNRatiosRefEE = laserAPDPNRatiosRefData.endcapItems();
        auto const& laserAlphasEE = laserAlphasData.endcapItems();
        auto const& linearCorrectionsEE = linearCorrectionsData.getValueMap().endcapItems();
        auto const& timeCalibConstantsEE = timeCalibConstantsData.endcapItems();

        const auto endcapSize = channelStatusData.endcapItems().size();
        for (unsigned int i = 0; i < endcapSize; ++i) {
          auto vi = view[barrelSize + i];

          vi.intercalibConstants() = intercalibConstantsEE[i];
          vi.channelStatus() = channelStatusEE[i].getEncodedStatusCode();

          vi.laserAPDPNRatios_p1() = laserAPDPNRatiosLaserEE[i].p1;
          vi.laserAPDPNRatios_p2() = laserAPDPNRatiosLaserEE[i].p2;
          vi.laserAPDPNRatios_p3() = laserAPDPNRatiosLaserEE[i].p3;

          vi.laserAPDPNref() = laserAPDPNRatiosRefEE[i];
          vi.laserAlpha() = laserAlphasEE[i];

          vi.linearCorrections_p1() = linearCorrectionsEE[i].p1;
          vi.linearCorrections_p2() = linearCorrectionsEE[i].p2;
          vi.linearCorrections_p3() = linearCorrectionsEE[i].p3;

          vi.timeCalibConstants() = timeCalibConstantsEE[i];
        }  // end Endcap loop

        // scalar data
        // ADC to GeV constants
        view.adcToGeVConstantEE() = adcToGeVConstantData.getEEValue();

        // time offset constants
        view.timeOffsetConstantEE() = timeOffsetConstantData.getEEValue();
      }

      // number of barrel items as offset for hashed ID access to EE items of columns
      view.offsetEE() = barrelSize;

      return product;
    }

  private:
    edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVConstantToken_;
    edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstantsToken_;
    edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelStatusToken_;
    edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> laserAPDPNRatiosToken_;
    edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> laserAPDPNRatiosRefToken_;
    edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> laserAlphasToken_;
    edm::ESGetToken<EcalLinearCorrections, EcalLinearCorrectionsRcd> linearCorrectionsToken_;
    edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> timeCalibConstantsToken_;
    edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> timeOffsetConstantToken_;

    bool const isPhase2_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(EcalRecHitConditionsHostESProducer);
