
/*----------------------------------------------------------------------

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

July 29, 2009       Added HcalValidationCorrs - Gena Kukartsev
September 21, 2009  Added HcalLutMetadata - Gena Kukartsev
   
----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/HcalAllRcds.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/HcalObjects/interface/AllObjects.h"

namespace edmtest {
  class HcalDumpConditions : public edm::one::EDAnalyzer<> {
  public:
    explicit HcalDumpConditions(edm::ParameterSet const& p) {
      front = p.getUntrackedParameter<std::string>("outFilePrefix", "Dump");
      mDumpRequest = p.getUntrackedParameter<std::vector<std::string> >("dump", std::vector<std::string>());
      m_toktopo = esConsumes<HcalTopology, HcalRecNumberingRecord>();
      m_tokdb = esConsumes<HcalDbService, HcalDbRecord>();

      tok_ElectronicsMap = esConsumes<HcalElectronicsMap, HcalElectronicsMapRcd>();
      tok_FrontEndMap = esConsumes<HcalFrontEndMap, HcalFrontEndMapRcd>();
      tok_QIEData = esConsumes<HcalQIEData, HcalQIEDataRcd>();
      tok_QIETypes = esConsumes<HcalQIETypes, HcalQIETypesRcd>();
      tok_Pedestals = esConsumes<HcalPedestals, HcalPedestalsRcd>();
      tok_PedestalWidths = esConsumes<HcalPedestalWidths, HcalPedestalWidthsRcd>();
      tok_Pedestals_effective = esConsumes<HcalPedestals, HcalPedestalsRcd>(edm::ESInputTag("", "effective"));
      tok_PedestalWidths_effective =
          esConsumes<HcalPedestalWidths, HcalPedestalWidthsRcd>(edm::ESInputTag("", "effective"));
      tok_Gains = esConsumes<HcalGains, HcalGainsRcd>();
      tok_GainWidths = esConsumes<HcalGainWidths, HcalGainWidthsRcd>();
      tok_PFCuts = esConsumes<HcalPFCuts, HcalPFCutsRcd>();
      tok_ChannelQuality = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>();
      tok_RespCorrs = esConsumes<HcalRespCorrs, HcalRespCorrsRcd>();
      tok_ZSThresholds = esConsumes<HcalZSThresholds, HcalZSThresholdsRcd>();
      tok_L1TriggerObjects = esConsumes<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd>();
      tok_TimeCorrs = esConsumes<HcalTimeCorrs, HcalTimeCorrsRcd>();
      tok_LUTCorrs = esConsumes<HcalLUTCorrs, HcalLUTCorrsRcd>();
      tok_PFCorrs = esConsumes<HcalPFCorrs, HcalPFCorrsRcd>();
      tok_ValidationCorrs = esConsumes<HcalValidationCorrs, HcalValidationCorrsRcd>();
      tok_LutMetadata = esConsumes<HcalLutMetadata, HcalLutMetadataRcd>();
      tok_DcsValues = esConsumes<HcalDcsValues, HcalDcsRcd>();
      tok_DcsMap = esConsumes<HcalDcsMap, HcalDcsMapRcd>();
      tok_RecoParams = esConsumes<HcalRecoParams, HcalRecoParamsRcd>();
      tok_TimingParams = esConsumes<HcalTimingParams, HcalTimingParamsRcd>();
      tok_LongRecoParams = esConsumes<HcalLongRecoParams, HcalLongRecoParamsRcd>();
      tok_ZDCLowGainFractions = esConsumes<HcalZDCLowGainFractions, HcalZDCLowGainFractionsRcd>();
      tok_MCParams = esConsumes<HcalMCParams, HcalMCParamsRcd>();
      tok_FlagHFDigiTimeParams = esConsumes<HcalFlagHFDigiTimeParams, HcalFlagHFDigiTimeParamsRcd>();
      tok_SiPMParameters = esConsumes<HcalSiPMParameters, HcalSiPMParametersRcd>();
      tok_SiPMCharacteristics = esConsumes<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd>();
      tok_TPChannelParameters = esConsumes<HcalTPChannelParameters, HcalTPChannelParametersRcd>();
      tok_TPParameters = esConsumes<HcalTPParameters, HcalTPParametersRcd>();
    }

    explicit HcalDumpConditions(int i) {}
    ~HcalDumpConditions() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    template <class S, class SRcd>
    void dumpIt(const std::vector<std::string>& mDumpRequest,
                const edm::Event& e,
                const edm::EventSetup& context,
                const std::string name,
                const edm::ESGetToken<S, SRcd> tok);
    template <class S, class SRcd>
    void dumpIt(const std::vector<std::string>& mDumpRequest,
                const edm::Event& e,
                const edm::EventSetup& context,
                const std::string name,
                const HcalTopology* topo,
                const edm::ESGetToken<S, SRcd> tok);
    template <class S>
    void writeToFile(const S& myS, const edm::Event& e, const std::string name);

  private:
    std::string front;
    std::vector<std::string> mDumpRequest;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> m_toktopo;
    edm::ESGetToken<HcalDbService, HcalDbRecord> m_tokdb;
    edm::ESGetToken<HcalElectronicsMap, HcalElectronicsMapRcd> tok_ElectronicsMap;
    edm::ESGetToken<HcalFrontEndMap, HcalFrontEndMapRcd> tok_FrontEndMap;
    edm::ESGetToken<HcalQIEData, HcalQIEDataRcd> tok_QIEData;
    edm::ESGetToken<HcalQIETypes, HcalQIETypesRcd> tok_QIETypes;
    edm::ESGetToken<HcalPedestals, HcalPedestalsRcd> tok_Pedestals;
    edm::ESGetToken<HcalPedestalWidths, HcalPedestalWidthsRcd> tok_PedestalWidths;
    edm::ESGetToken<HcalPedestals, HcalPedestalsRcd> tok_Pedestals_effective;
    edm::ESGetToken<HcalPedestalWidths, HcalPedestalWidthsRcd> tok_PedestalWidths_effective;
    edm::ESGetToken<HcalGains, HcalGainsRcd> tok_Gains;
    edm::ESGetToken<HcalGainWidths, HcalGainWidthsRcd> tok_GainWidths;
    edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> tok_PFCuts;
    edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> tok_ChannelQuality;
    edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_RespCorrs;
    edm::ESGetToken<HcalZSThresholds, HcalZSThresholdsRcd> tok_ZSThresholds;
    edm::ESGetToken<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd> tok_L1TriggerObjects;
    edm::ESGetToken<HcalTimeCorrs, HcalTimeCorrsRcd> tok_TimeCorrs;
    edm::ESGetToken<HcalLUTCorrs, HcalLUTCorrsRcd> tok_LUTCorrs;
    edm::ESGetToken<HcalPFCorrs, HcalPFCorrsRcd> tok_PFCorrs;
    edm::ESGetToken<HcalValidationCorrs, HcalValidationCorrsRcd> tok_ValidationCorrs;
    edm::ESGetToken<HcalLutMetadata, HcalLutMetadataRcd> tok_LutMetadata;
    edm::ESGetToken<HcalDcsValues, HcalDcsRcd> tok_DcsValues;
    edm::ESGetToken<HcalDcsMap, HcalDcsMapRcd> tok_DcsMap;
    edm::ESGetToken<HcalRecoParams, HcalRecoParamsRcd> tok_RecoParams;
    edm::ESGetToken<HcalTimingParams, HcalTimingParamsRcd> tok_TimingParams;
    edm::ESGetToken<HcalLongRecoParams, HcalLongRecoParamsRcd> tok_LongRecoParams;
    edm::ESGetToken<HcalZDCLowGainFractions, HcalZDCLowGainFractionsRcd> tok_ZDCLowGainFractions;
    edm::ESGetToken<HcalMCParams, HcalMCParamsRcd> tok_MCParams;
    edm::ESGetToken<HcalFlagHFDigiTimeParams, HcalFlagHFDigiTimeParamsRcd> tok_FlagHFDigiTimeParams;
    edm::ESGetToken<HcalSiPMParameters, HcalSiPMParametersRcd> tok_SiPMParameters;
    edm::ESGetToken<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd> tok_SiPMCharacteristics;
    edm::ESGetToken<HcalTPChannelParameters, HcalTPChannelParametersRcd> tok_TPChannelParameters;
    edm::ESGetToken<HcalTPParameters, HcalTPParametersRcd> tok_TPParameters;
  };

  template <class S, class SRcd>
  void HcalDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name,
                                  const edm::ESGetToken<S, SRcd> tok) {
    if (std::find(mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end()) {
      const S& myobject = context.getData(tok);

      writeToFile(myobject, e, name);

      if (context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue())
        std::cout << "error: invalid IOV sync value !" << std::endl;
    }
  }

  template <class S, class SRcd>
  void HcalDumpConditions::dumpIt(const std::vector<std::string>& mDumpRequest,
                                  const edm::Event& e,
                                  const edm::EventSetup& context,
                                  const std::string name,
                                  const HcalTopology* topo,
                                  const edm::ESGetToken<S, SRcd> tok) {
    if (std::find(mDumpRequest.begin(), mDumpRequest.end(), name) != mDumpRequest.end()) {
      S myobject = context.getData(tok);
      if (topo)
        myobject.setTopo(topo);

      writeToFile(myobject, e, name);

      if (context.get<SRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue())
        std::cout << "error: invalid IOV sync value !" << std::endl;
    }
  }

  template <class S>
  void HcalDumpConditions::writeToFile(const S& myS, const edm::Event& e, const std::string name) {
    int myrun = e.id().run();
    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str());
    std::cout << "HcalDumpConditions: ---- Dumping " << name << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject(outStream, myS);
  }

  void HcalDumpConditions::analyze(const edm::Event& e, const edm::EventSetup& context) {
    const HcalTopology* topo = &context.getData(m_toktopo);

    using namespace edm::eventsetup;
    std::cout << "HcalDumpConditions::analyze-> I AM IN RUN NUMBER " << e.id().run() << std::endl;

    if (mDumpRequest.empty())
      return;

    // dumpIt called for all possible ValueMaps. The function checks if the dump is actually requested.

    dumpIt<HcalElectronicsMap, HcalElectronicsMapRcd>(mDumpRequest, e, context, "ElectronicsMap", tok_ElectronicsMap);
    dumpIt<HcalFrontEndMap, HcalFrontEndMapRcd>(mDumpRequest, e, context, "FrontEndMap", tok_FrontEndMap);
    dumpIt<HcalQIEData, HcalQIEDataRcd>(mDumpRequest, e, context, "QIEData", topo, tok_QIEData);
    dumpIt<HcalQIETypes, HcalQIETypesRcd>(mDumpRequest, e, context, "QIETypes", topo, tok_QIETypes);
    dumpIt<HcalPedestals, HcalPedestalsRcd>(mDumpRequest, e, context, "Pedestals", topo, tok_Pedestals);
    dumpIt<HcalPedestalWidths, HcalPedestalWidthsRcd>(
        mDumpRequest, e, context, "PedestalWidths", topo, tok_PedestalWidths);
    dumpIt<HcalPedestals, HcalPedestalsRcd>(
        mDumpRequest, e, context, "EffectivePedestals", topo, tok_Pedestals_effective);
    dumpIt<HcalPedestalWidths, HcalPedestalWidthsRcd>(
        mDumpRequest, e, context, "EffectivePedestalWidths", topo, tok_PedestalWidths_effective);
    dumpIt<HcalGains, HcalGainsRcd>(mDumpRequest, e, context, "Gains", topo, tok_Gains);
    dumpIt<HcalGainWidths, HcalGainWidthsRcd>(mDumpRequest, e, context, "GainWidths", topo, tok_GainWidths);
    dumpIt<HcalPFCuts, HcalPFCutsRcd>(mDumpRequest, e, context, "PFCuts", topo, tok_PFCuts);
    dumpIt<HcalChannelQuality, HcalChannelQualityRcd>(
        mDumpRequest, e, context, "ChannelQuality", topo, tok_ChannelQuality);
    dumpIt<HcalRespCorrs, HcalRespCorrsRcd>(mDumpRequest, e, context, "RespCorrs", topo, tok_RespCorrs);
    dumpIt<HcalZSThresholds, HcalZSThresholdsRcd>(mDumpRequest, e, context, "ZSThresholds", topo, tok_ZSThresholds);
    dumpIt<HcalL1TriggerObjects, HcalL1TriggerObjectsRcd>(
        mDumpRequest, e, context, "L1TriggerObjects", topo, tok_L1TriggerObjects);
    dumpIt<HcalTimeCorrs, HcalTimeCorrsRcd>(mDumpRequest, e, context, "TimeCorrs", topo, tok_TimeCorrs);
    dumpIt<HcalLUTCorrs, HcalLUTCorrsRcd>(mDumpRequest, e, context, "LUTCorrs", topo, tok_LUTCorrs);
    dumpIt<HcalPFCorrs, HcalPFCorrsRcd>(mDumpRequest, e, context, "PFCorrs", topo, tok_PFCorrs);
    dumpIt<HcalValidationCorrs, HcalValidationCorrsRcd>(
        mDumpRequest, e, context, "ValidationCorrs", topo, tok_ValidationCorrs);
    dumpIt<HcalLutMetadata, HcalLutMetadataRcd>(mDumpRequest, e, context, "LutMetadata", topo, tok_LutMetadata);
    dumpIt<HcalDcsValues, HcalDcsRcd>(mDumpRequest, e, context, "DcsValues", tok_DcsValues);
    dumpIt<HcalDcsMap, HcalDcsMapRcd>(mDumpRequest, e, context, "DcsMap", tok_DcsMap);
    dumpIt<HcalRecoParams, HcalRecoParamsRcd>(mDumpRequest, e, context, "RecoParams", topo, tok_RecoParams);
    dumpIt<HcalTimingParams, HcalTimingParamsRcd>(mDumpRequest, e, context, "TimingParams", topo, tok_TimingParams);
    dumpIt<HcalLongRecoParams, HcalLongRecoParamsRcd>(
        mDumpRequest, e, context, "LongRecoParams", topo, tok_LongRecoParams);
    dumpIt<HcalZDCLowGainFractions, HcalZDCLowGainFractionsRcd>(
        mDumpRequest, e, context, "ZDCLowGainFractions", topo, tok_ZDCLowGainFractions);
    dumpIt<HcalMCParams, HcalMCParamsRcd>(mDumpRequest, e, context, "MCParams", topo, tok_MCParams);
    dumpIt<HcalFlagHFDigiTimeParams, HcalFlagHFDigiTimeParamsRcd>(
        mDumpRequest, e, context, "FlagHFDigiTimeParams", topo, tok_FlagHFDigiTimeParams);
    dumpIt<HcalSiPMParameters, HcalSiPMParametersRcd>(
        mDumpRequest, e, context, "SiPMParameters", topo, tok_SiPMParameters);
    dumpIt<HcalSiPMCharacteristics, HcalSiPMCharacteristicsRcd>(
        mDumpRequest, e, context, "SiPMCharacteristics", tok_SiPMCharacteristics);
    dumpIt<HcalTPChannelParameters, HcalTPChannelParametersRcd>(
        mDumpRequest, e, context, "TPChannelParameters", topo, tok_TPChannelParameters);
    dumpIt<HcalTPParameters, HcalTPParametersRcd>(mDumpRequest, e, context, "TPParameters", tok_TPParameters);
  }
  DEFINE_FWK_MODULE(HcalDumpConditions);
}  // namespace edmtest
