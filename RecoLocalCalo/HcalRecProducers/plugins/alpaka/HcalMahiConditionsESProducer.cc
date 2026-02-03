#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "CondFormats/HcalObjects/interface/alpaka/HcalMahiConditionsDevice.h"
#include "CondFormats/HcalObjects/interface/HcalMahiConditionsSoA.h"
#include "CondFormats/DataRecord/interface/HcalMahiConditionsRcd.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace {
    float convertPedWidths(
        float const value, float const width, int const i, HcalQIECoder const& coder, HcalQIEShape const& shape) {
      float const y = value;
      float const x = width;
      unsigned const x1 = static_cast<unsigned>(std::floor(y));
      unsigned const x2 = static_cast<unsigned>(std::floor(y + 1.));
      unsigned iun = static_cast<unsigned>(i);
      float const y1 = coder.charge(shape, x1, iun);
      float const y2 = coder.charge(shape, x2, iun);
      return (y2 - y1) * x;
    }
    float convertPed(float const x, int const i, HcalQIECoder const& coder, HcalQIEShape const& shape) {
      int const x1 = static_cast<int>(std::floor(x));
      int const x2 = static_cast<int>(std::floor(x + 1));
      float const y2 = coder.charge(shape, x2, i);
      float const y1 = coder.charge(shape, x1, i);
      return (y2 - y1) * (x - x1) + y1;
    }
  }  // namespace

  class HcalMahiConditionsESProducer : public ESProducer {
  public:
    HcalMahiConditionsESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      recoParamsToken_ = cc.consumes();
      pedestalsToken_ = cc.consumes();
      effectivePedestalsToken_ = cc.consumes(edm::ESInputTag{"", "withTopoEff"});
      gainsToken_ = cc.consumes();
      lutCorrsToken_ = cc.consumes();
      respCorrsToken_ = cc.consumes();
      timeCorrsToken_ = cc.consumes();
      pedestalWidthsToken_ = cc.consumes();
      effectivePedestalWidthsToken_ = cc.consumes(edm::ESInputTag{"", "withTopoEff"});
      gainWidthsToken_ = cc.consumes();
      channelQualityToken_ = cc.consumes();
      qieTypesToken_ = cc.consumes();
      qieDataToken_ = cc.consumes();
      sipmParametersToken_ = cc.consumes();
      topologyToken_ = cc.consumes();
      recConstantsToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<hcal::HcalMahiConditionsPortableHost> produce(HcalMahiConditionsRcd const& iRecord) {
      auto const& recoParams = iRecord.get(recoParamsToken_);
      auto const& pedestals = iRecord.get(pedestalsToken_);
      auto const& effectivePedestals = iRecord.get(effectivePedestalsToken_);
      auto const& gains = iRecord.get(gainsToken_);
      auto const& lutCorrs = iRecord.get(lutCorrsToken_);
      auto const& respCorrs = iRecord.get(respCorrsToken_);
      auto const& timeCorrs = iRecord.get(timeCorrsToken_);
      auto const& pedestalWidths = iRecord.get(pedestalWidthsToken_);
      auto const& effectivePedestalWidths = iRecord.get(effectivePedestalWidthsToken_);
      auto const& gainWidths = iRecord.get(gainWidthsToken_);
      auto const& channelQuality = iRecord.get(channelQualityToken_);
      auto const& qieTypes = iRecord.get(qieTypesToken_);
      auto const& qieData = iRecord.get(qieDataToken_);
      auto const& sipmParameters = iRecord.get(sipmParametersToken_);
      auto const& topology = iRecord.get(topologyToken_);
      auto const& recConstants = iRecord.get(recConstantsToken_);

      size_t const totalChannels =
          pedestals.getAllContainers()[0].second.size() + pedestals.getAllContainers()[1].second.size();

      auto product = std::make_unique<hcal::HcalMahiConditionsPortableHost>(cms::alpakatools::host(), totalChannels);

      auto view = product->view();

      // convert pedestals
      auto const ped_unitIsADC = pedestals.isADC();
      auto const effPed_unitIsADC = effectivePedestals.isADC();

      // fill HB channels
      auto const recoParams_containers = recoParams.getAllContainers();
      auto const pedestals_containers = pedestals.getAllContainers();
      auto const effectivePedestals_containers = effectivePedestals.getAllContainers();
      auto const gains_containers = gains.getAllContainers();
      auto const lutCorrs_containers = lutCorrs.getAllContainers();
      auto const respCorrs_containers = respCorrs.getAllContainers();
      auto const timeCorrs_containers = timeCorrs.getAllContainers();
      auto const pedestalWidths_containers = pedestalWidths.getAllContainers();
      auto const effectivePedestalWidths_containers = effectivePedestalWidths.getAllContainers();
      auto const gainWidths_containers = gainWidths.getAllContainers();
      auto const channelQuality_containers = channelQuality.getAllContainers();
      auto const qieTypes_containers = qieTypes.getAllContainers();
      auto const qieData_containers = qieData.getAllContainers();
      auto const sipmParameters_containers = sipmParameters.getAllContainers();

      auto const& pedestals_barrel = pedestals_containers[0].second;
      auto const& effectivePedestals_barrel = effectivePedestals_containers[0].second;
      auto const& gains_barrel = gains_containers[0].second;
      auto const& lutCorrs_barrel = lutCorrs_containers[0].second;
      auto const& respCorrs_barrel = respCorrs_containers[0].second;
      auto const& timeCorrs_barrel = timeCorrs_containers[0].second;
      auto const& pedestalWidths_barrel = pedestalWidths_containers[0].second;
      auto const& effectivePedestalWidths_barrel = effectivePedestalWidths_containers[0].second;
      auto const& gainWidths_barrel = gainWidths_containers[0].second;
      auto const& channelQuality_barrel = channelQuality_containers[0].second;
      auto const& qieTypes_barrel = qieTypes_containers[0].second;
      auto const& qieData_barrel = qieData_containers[0].second;
      auto const& sipmParameters_barrel = sipmParameters_containers[0].second;

      for (uint64_t i = 0; i < pedestals_barrel.size(); ++i) {
        auto vi = view[i];

        // convert pedestals
        auto const& qieCoder = qieData_barrel[i];
        auto const qieType = qieTypes_barrel[i].getValue() > 1 ? 1 : 0;
        auto const& qieShape = qieData.getShape(qieType);

        // covert pedestal values if unit is ADC
        vi.pedestals_value()[0] = ped_unitIsADC ? convertPed(pedestals_barrel[i].getValue(0), 0, qieCoder, qieShape)
                                                : pedestals_barrel[i].getValue(0);
        vi.pedestals_value()[1] = ped_unitIsADC ? convertPed(pedestals_barrel[i].getValue(1), 1, qieCoder, qieShape)
                                                : pedestals_barrel[i].getValue(1);
        vi.pedestals_value()[2] = ped_unitIsADC ? convertPed(pedestals_barrel[i].getValue(2), 2, qieCoder, qieShape)
                                                : pedestals_barrel[i].getValue(2);
        vi.pedestals_value()[3] = ped_unitIsADC ? convertPed(pedestals_barrel[i].getValue(3), 3, qieCoder, qieShape)
                                                : pedestals_barrel[i].getValue(3);

        vi.pedestals_width()[0] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_barrel[i].getValue(0), pedestalWidths_barrel[i].getWidth(0), 0, qieCoder, qieShape)
                : pedestalWidths_barrel[i].getWidth(0);
        vi.pedestals_width()[1] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_barrel[i].getValue(1), pedestalWidths_barrel[i].getWidth(1), 1, qieCoder, qieShape)
                : pedestalWidths_barrel[i].getWidth(1);
        vi.pedestals_width()[2] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_barrel[i].getValue(2), pedestalWidths_barrel[i].getWidth(2), 2, qieCoder, qieShape)
                : pedestalWidths_barrel[i].getWidth(2);
        vi.pedestals_width()[3] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_barrel[i].getValue(3), pedestalWidths_barrel[i].getWidth(3), 3, qieCoder, qieShape)
                : pedestalWidths_barrel[i].getWidth(3);

        vi.effectivePedestals()[0] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_barrel[i].getValue(0), 0, qieCoder, qieShape)
                                         : effectivePedestals_barrel[i].getValue(0);
        vi.effectivePedestals()[1] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_barrel[i].getValue(1), 1, qieCoder, qieShape)
                                         : effectivePedestals_barrel[i].getValue(1);
        vi.effectivePedestals()[2] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_barrel[i].getValue(2), 2, qieCoder, qieShape)
                                         : effectivePedestals_barrel[i].getValue(2);
        vi.effectivePedestals()[3] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_barrel[i].getValue(3), 3, qieCoder, qieShape)
                                         : effectivePedestals_barrel[i].getValue(3);

        vi.effectivePedestalWidths()[0] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_barrel[i].getValue(0),
                                                                 effectivePedestalWidths_barrel[i].getWidth(0),
                                                                 0,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_barrel[i].getWidth(0);
        vi.effectivePedestalWidths()[1] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_barrel[i].getValue(1),
                                                                 effectivePedestalWidths_barrel[i].getWidth(1),
                                                                 1,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_barrel[i].getWidth(1);
        vi.effectivePedestalWidths()[2] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_barrel[i].getValue(2),
                                                                 effectivePedestalWidths_barrel[i].getWidth(2),
                                                                 2,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_barrel[i].getWidth(2);
        vi.effectivePedestalWidths()[3] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_barrel[i].getValue(3),
                                                                 effectivePedestalWidths_barrel[i].getWidth(3),
                                                                 3,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_barrel[i].getWidth(3);

        vi.gains_value()[0] = gains_barrel[i].getValue(0);
        vi.gains_value()[1] = gains_barrel[i].getValue(1);
        vi.gains_value()[2] = gains_barrel[i].getValue(2);
        vi.gains_value()[3] = gains_barrel[i].getValue(3);

        vi.lutCorrs_values() = lutCorrs_barrel[i].getValue();
        vi.respCorrs_values() = respCorrs_barrel[i].getValue();
        vi.timeCorrs_values() = timeCorrs_barrel[i].getValue();

        vi.pedestalWidths_sigma00() = *(pedestalWidths_barrel[i].getValues());
        vi.pedestalWidths_sigma01() = *(pedestalWidths_barrel[i].getValues() + 1);
        vi.pedestalWidths_sigma02() = *(pedestalWidths_barrel[i].getValues() + 2);
        vi.pedestalWidths_sigma03() = *(pedestalWidths_barrel[i].getValues() + 3);
        vi.pedestalWidths_sigma10() = *(pedestalWidths_barrel[i].getValues() + 4);
        vi.pedestalWidths_sigma11() = *(pedestalWidths_barrel[i].getValues() + 5);
        vi.pedestalWidths_sigma12() = *(pedestalWidths_barrel[i].getValues() + 6);
        vi.pedestalWidths_sigma13() = *(pedestalWidths_barrel[i].getValues() + 7);
        vi.pedestalWidths_sigma20() = *(pedestalWidths_barrel[i].getValues() + 8);
        vi.pedestalWidths_sigma21() = *(pedestalWidths_barrel[i].getValues() + 9);
        vi.pedestalWidths_sigma22() = *(pedestalWidths_barrel[i].getValues() + 10);
        vi.pedestalWidths_sigma23() = *(pedestalWidths_barrel[i].getValues() + 11);
        vi.pedestalWidths_sigma30() = *(pedestalWidths_barrel[i].getValues() + 12);
        vi.pedestalWidths_sigma31() = *(pedestalWidths_barrel[i].getValues() + 13);
        vi.pedestalWidths_sigma32() = *(pedestalWidths_barrel[i].getValues() + 14);
        vi.pedestalWidths_sigma33() = *(pedestalWidths_barrel[i].getValues() + 15);

        vi.gainWidths_value0() = gainWidths_barrel[i].getValue(0);
        vi.gainWidths_value1() = gainWidths_barrel[i].getValue(1);
        vi.gainWidths_value2() = gainWidths_barrel[i].getValue(2);
        vi.gainWidths_value3() = gainWidths_barrel[i].getValue(3);

        vi.channelQuality_status() = channelQuality_barrel[i].getValue();
        vi.qieTypes_values() = qieTypes_barrel[i].getValue();

        for (uint32_t k = 0; k < 4; k++)
          for (uint32_t l = 0; l < 4; l++) {
            auto const linear = k * 4 + l;
            vi.qieCoders_offsets()[linear] = qieData_barrel[i].offset(k, l);
            vi.qieCoders_slopes()[linear] = qieData_barrel[i].slope(k, l);
          }

        vi.sipmPar_type() = sipmParameters_barrel[i].getType();
        vi.sipmPar_auxi1() = sipmParameters_barrel[i].getauxi1();
        vi.sipmPar_fcByPE() = sipmParameters_barrel[i].getFCByPE();
        vi.sipmPar_darkCurrent() = sipmParameters_barrel[i].getDarkCurrent();
        vi.sipmPar_auxi2() = sipmParameters_barrel[i].getauxi2();
      }

      // fill HE channels
      auto const& pedestals_endcaps = pedestals_containers[1].second;
      auto const& effectivePedestals_endcaps = effectivePedestals_containers[1].second;
      auto const& gains_endcaps = gains_containers[1].second;
      auto const& lutCorrs_endcaps = lutCorrs_containers[1].second;
      auto const& respCorrs_endcaps = respCorrs_containers[1].second;
      auto const& timeCorrs_endcaps = timeCorrs_containers[1].second;
      auto const& pedestalWidths_endcaps = pedestalWidths_containers[1].second;
      auto const& effectivePedestalWidths_endcaps = effectivePedestalWidths_containers[1].second;
      auto const& gainWidths_endcaps = gainWidths_containers[1].second;
      auto const& channelQuality_endcaps = channelQuality_containers[1].second;
      auto const& qieTypes_endcaps = qieTypes_containers[1].second;
      auto const& qieData_endcaps = qieData_containers[1].second;
      auto const& sipmParameters_endcaps = sipmParameters_containers[1].second;

      auto const offset = pedestals_barrel.size();

      for (uint64_t i = 0; i < pedestals_endcaps.size(); ++i) {
        auto const& qieCoder = qieData_endcaps[i];
        auto const qieType = qieTypes_endcaps[i].getValue() > 1 ? 1 : 0;
        auto const& qieShape = qieData.getShape(qieType);

        auto vi = view[offset + i];

        vi.pedestals_value()[0] = ped_unitIsADC ? convertPed(pedestals_endcaps[i].getValue(0), 0, qieCoder, qieShape)
                                                : pedestals_endcaps[i].getValue(0);
        vi.pedestals_value()[1] = ped_unitIsADC ? convertPed(pedestals_endcaps[i].getValue(1), 1, qieCoder, qieShape)
                                                : pedestals_endcaps[i].getValue(1);
        vi.pedestals_value()[2] = ped_unitIsADC ? convertPed(pedestals_endcaps[i].getValue(2), 2, qieCoder, qieShape)
                                                : pedestals_endcaps[i].getValue(2);
        vi.pedestals_value()[3] = ped_unitIsADC ? convertPed(pedestals_endcaps[i].getValue(3), 3, qieCoder, qieShape)
                                                : pedestals_endcaps[i].getValue(3);

        vi.pedestals_width()[0] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_endcaps[i].getValue(0), pedestalWidths_endcaps[i].getWidth(0), 0, qieCoder, qieShape)
                : pedestalWidths_endcaps[i].getWidth(0);
        vi.pedestals_width()[1] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_endcaps[i].getValue(1), pedestalWidths_endcaps[i].getWidth(1), 1, qieCoder, qieShape)
                : pedestalWidths_endcaps[i].getWidth(1);
        vi.pedestals_width()[2] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_endcaps[i].getValue(2), pedestalWidths_endcaps[i].getWidth(2), 2, qieCoder, qieShape)
                : pedestalWidths_endcaps[i].getWidth(2);
        vi.pedestals_width()[3] =
            ped_unitIsADC
                ? convertPedWidths(
                      pedestals_endcaps[i].getValue(3), pedestalWidths_endcaps[i].getWidth(3), 3, qieCoder, qieShape)
                : pedestalWidths_endcaps[i].getWidth(3);

        vi.effectivePedestals()[0] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_endcaps[i].getValue(0), 0, qieCoder, qieShape)
                                         : effectivePedestals_endcaps[i].getValue(0);
        vi.effectivePedestals()[1] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_endcaps[i].getValue(1), 1, qieCoder, qieShape)
                                         : effectivePedestals_endcaps[i].getValue(1);
        vi.effectivePedestals()[2] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_endcaps[i].getValue(2), 2, qieCoder, qieShape)
                                         : effectivePedestals_endcaps[i].getValue(2);
        vi.effectivePedestals()[3] = effPed_unitIsADC
                                         ? convertPed(effectivePedestals_endcaps[i].getValue(3), 3, qieCoder, qieShape)
                                         : effectivePedestals_endcaps[i].getValue(3);

        vi.effectivePedestalWidths()[0] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_endcaps[i].getValue(0),
                                                                 effectivePedestalWidths_endcaps[i].getWidth(0),
                                                                 0,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_endcaps[i].getWidth(0);
        vi.effectivePedestalWidths()[1] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_endcaps[i].getValue(1),
                                                                 effectivePedestalWidths_endcaps[i].getWidth(1),
                                                                 1,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_endcaps[i].getWidth(1);
        vi.effectivePedestalWidths()[2] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_endcaps[i].getValue(2),
                                                                 effectivePedestalWidths_endcaps[i].getWidth(2),
                                                                 2,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_endcaps[i].getWidth(2);
        vi.effectivePedestalWidths()[3] = effPed_unitIsADC
                                              ? convertPedWidths(effectivePedestals_endcaps[i].getValue(3),
                                                                 effectivePedestalWidths_endcaps[i].getWidth(3),
                                                                 3,
                                                                 qieCoder,
                                                                 qieShape)
                                              : effectivePedestalWidths_endcaps[i].getWidth(3);

        vi.gains_value()[0] = gains_endcaps[i].getValue(0);
        vi.gains_value()[1] = gains_endcaps[i].getValue(1);
        vi.gains_value()[2] = gains_endcaps[i].getValue(2);
        vi.gains_value()[3] = gains_endcaps[i].getValue(3);

        vi.lutCorrs_values() = lutCorrs_endcaps[i].getValue();
        vi.respCorrs_values() = respCorrs_endcaps[i].getValue();
        vi.timeCorrs_values() = timeCorrs_endcaps[i].getValue();

        vi.pedestalWidths_sigma00() = *(pedestalWidths_endcaps[i].getValues());
        vi.pedestalWidths_sigma01() = *(pedestalWidths_endcaps[i].getValues() + 1);
        vi.pedestalWidths_sigma02() = *(pedestalWidths_endcaps[i].getValues() + 2);
        vi.pedestalWidths_sigma03() = *(pedestalWidths_endcaps[i].getValues() + 3);
        vi.pedestalWidths_sigma10() = *(pedestalWidths_endcaps[i].getValues() + 4);
        vi.pedestalWidths_sigma11() = *(pedestalWidths_endcaps[i].getValues() + 5);
        vi.pedestalWidths_sigma12() = *(pedestalWidths_endcaps[i].getValues() + 6);
        vi.pedestalWidths_sigma13() = *(pedestalWidths_endcaps[i].getValues() + 7);
        vi.pedestalWidths_sigma20() = *(pedestalWidths_endcaps[i].getValues() + 8);
        vi.pedestalWidths_sigma21() = *(pedestalWidths_endcaps[i].getValues() + 9);
        vi.pedestalWidths_sigma22() = *(pedestalWidths_endcaps[i].getValues() + 10);
        vi.pedestalWidths_sigma23() = *(pedestalWidths_endcaps[i].getValues() + 11);
        vi.pedestalWidths_sigma30() = *(pedestalWidths_endcaps[i].getValues() + 12);
        vi.pedestalWidths_sigma31() = *(pedestalWidths_endcaps[i].getValues() + 13);
        vi.pedestalWidths_sigma32() = *(pedestalWidths_endcaps[i].getValues() + 14);
        vi.pedestalWidths_sigma33() = *(pedestalWidths_endcaps[i].getValues() + 15);

        vi.gainWidths_value0() = gainWidths_endcaps[i].getValue(0);
        vi.gainWidths_value1() = gainWidths_endcaps[i].getValue(1);
        vi.gainWidths_value2() = gainWidths_endcaps[i].getValue(2);
        vi.gainWidths_value3() = gainWidths_endcaps[i].getValue(3);

        vi.channelQuality_status() = channelQuality_endcaps[i].getValue();
        vi.qieTypes_values() = qieTypes_endcaps[i].getValue();

        for (uint32_t k = 0; k < 4; k++)
          for (uint32_t l = 0; l < 4; l++) {
            auto const linear = k * 4u + l;
            vi.qieCoders_offsets()[linear] = qieData_endcaps[i].offset(k, l);
            vi.qieCoders_slopes()[linear] = qieData_endcaps[i].slope(k, l);
          }

        vi.sipmPar_type() = sipmParameters_endcaps[i].getType();
        vi.sipmPar_auxi1() = sipmParameters_endcaps[i].getauxi1();
        vi.sipmPar_fcByPE() = sipmParameters_endcaps[i].getFCByPE();
        vi.sipmPar_darkCurrent() = sipmParameters_endcaps[i].getDarkCurrent();
        vi.sipmPar_auxi2() = sipmParameters_endcaps[i].getauxi2();
      }
      //fill the scalars
      static const int IPHI_MAX = 72;  // private member of topology

      view.maxDepthHB() = topology.maxDepthHB();
      view.maxDepthHE() = topology.maxDepthHE();
      view.maxPhiHE() = recConstants.getNPhi(1) > IPHI_MAX ? recConstants.getNPhi(1) : IPHI_MAX;
      view.firstHBRing() = topology.firstHBRing();
      view.lastHBRing() = topology.lastHBRing();
      view.firstHERing() = topology.firstHERing();
      view.lastHERing() = topology.lastHERing();
      view.nEtaHB() = recConstants.getEtaRange(0).second - recConstants.getEtaRange(0).first + 1;
      view.nEtaHE() =
          topology.firstHERing() > topology.lastHERing() ? 0 : (topology.lastHERing() - topology.firstHERing() + 1);
      view.offsetForHashes() = offset;

      return product;
    }

  private:
    edm::ESGetToken<HcalRecoParams, HcalRecoParamsRcd> recoParamsToken_;
    edm::ESGetToken<HcalPedestals, HcalPedestalsRcd> pedestalsToken_;
    edm::ESGetToken<HcalPedestals, HcalPedestalsRcd> effectivePedestalsToken_;
    edm::ESGetToken<HcalGains, HcalGainsRcd> gainsToken_;
    edm::ESGetToken<HcalLUTCorrs, HcalLUTCorrsRcd> lutCorrsToken_;
    edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> respCorrsToken_;
    edm::ESGetToken<HcalTimeCorrs, HcalTimeCorrsRcd> timeCorrsToken_;
    edm::ESGetToken<HcalPedestalWidths, HcalPedestalWidthsRcd> pedestalWidthsToken_;
    edm::ESGetToken<HcalPedestalWidths, HcalPedestalWidthsRcd> effectivePedestalWidthsToken_;
    edm::ESGetToken<HcalGainWidths, HcalGainWidthsRcd> gainWidthsToken_;
    edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> channelQualityToken_;
    edm::ESGetToken<HcalQIETypes, HcalQIETypesRcd> qieTypesToken_;
    edm::ESGetToken<HcalQIEData, HcalQIEDataRcd> qieDataToken_;
    edm::ESGetToken<HcalSiPMParameters, HcalSiPMParametersRcd> sipmParametersToken_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topologyToken_;
    edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> recConstantsToken_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(HcalMahiConditionsESProducer);
