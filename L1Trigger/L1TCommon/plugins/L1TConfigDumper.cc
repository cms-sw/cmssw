
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"

#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"

#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
//#include "CondFormats/L1TObjects/interface/L1JetEtScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
//#include "CondFormats/L1TObjects/interface/L1EmEtScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
//#include "CondFormats/L1TObjects/interface/L1HtMissScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
//#include "CondFormats/L1TObjects/interface/L1HfRingEtScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"

#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"

#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"

#include "CondFormats/DataRecord/interface/L1MuCSCTFAlignmentRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFAlignment.h"

#include "CondFormats/DataRecord/interface/L1MuCSCPtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTEtaPatternLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTExtLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTPhiLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTPtaLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTQualPatternLutRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"

#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"

#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"

#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"

#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"

#include "CondFormats/DataRecord/interface/L1RCTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"

#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"

#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"

#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"

#include "CondFormats/DataRecord/interface/L1GtParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"

#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoAlgoTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMaskVetoAlgoTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "CondFormats/DataRecord/interface/L1GtPsbSetupRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtPsbSetup.h"

#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"  //Record spelled out
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"

#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"

#include "CondFormats/DataRecord/interface/L1RCTChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTChannelMask.h"

#include "CondFormats/DataRecord/interface/L1RCTNoisyChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1RCTNoisyChannelMask.h"

#include "CondFormats/DataRecord/interface/L1GctChannelMaskRcd.h"
#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactorsAlgoTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactorsTechTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMaskAlgoTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMaskTechTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

#include "CondFormats/DataRecord/interface/L1GtTriggerMaskVetoTechTrigRcd.h"
//#include "CondFormats/L1TObjects/interface/L1GtTriggerMaskVetoTechTrig.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"

//#include "CondFormats/DataRecord/interface/NumL1CondRcd.h"
//#include "CondFormats/L1TObjects/interface/NumL1Cond.h"

class L1TConfigDumper : public edm::one::EDAnalyzer<> {
public:
  explicit L1TConfigDumper(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1TriggerKeyList, L1TriggerKeyListRcd> AToken_;
  edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd> BToken_;
  edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> CToken_;
  edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> DToken_;
  edm::ESGetToken<L1CaloEtScale, L1HtMissScaleRcd> EToken_;
  edm::ESGetToken<L1CaloEtScale, L1HfRingEtScaleRcd> FToken_;
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> GToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> HToken_;
  edm::ESGetToken<L1MuGMTScales, L1MuGMTScalesRcd> IToken_;
  edm::ESGetToken<L1MuCSCTFConfiguration, L1MuCSCTFConfigurationRcd> JToken_;
  edm::ESGetToken<L1MuCSCTFAlignment, L1MuCSCTFAlignmentRcd> KToken_;
  edm::ESGetToken<L1MuCSCPtLut, L1MuCSCPtLutRcd> LToken_;
  edm::ESGetToken<L1MuDTEtaPatternLut, L1MuDTEtaPatternLutRcd> MToken_;
  edm::ESGetToken<L1MuDTExtLut, L1MuDTExtLutRcd> NToken_;
  edm::ESGetToken<L1MuDTPhiLut, L1MuDTPhiLutRcd> OToken_;
  edm::ESGetToken<L1MuDTPtaLut, L1MuDTPtaLutRcd> PToken_;
  edm::ESGetToken<L1MuDTQualPatternLut, L1MuDTQualPatternLutRcd> QToken_;
  edm::ESGetToken<L1MuDTTFParameters, L1MuDTTFParametersRcd> RToken_;
  edm::ESGetToken<L1RPCConfig, L1RPCConfigRcd> SToken_;
  edm::ESGetToken<L1RPCConeDefinition, L1RPCConeDefinitionRcd> TToken_;
  edm::ESGetToken<L1RPCHsbConfig, L1RPCHsbConfigRcd> UToken_;
  edm::ESGetToken<L1RPCBxOrConfig, L1RPCBxOrConfigRcd> VToken_;
  edm::ESGetToken<L1MuGMTParameters, L1MuGMTParametersRcd> WToken_;
  edm::ESGetToken<L1RCTParameters, L1RCTParametersRcd> XToken_;
  edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> YToken_;
  edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> ZToken_;
  edm::ESGetToken<L1GctJetFinderParams, L1GctJetFinderParamsRcd> AAToken_;
  edm::ESGetToken<L1GtBoardMaps, L1GtBoardMapsRcd> BBToken_;
  edm::ESGetToken<L1GtParameters, L1GtParametersRcd> CCToken_;
  edm::ESGetToken<L1GtStableParameters, L1GtStableParametersRcd> DDToken_;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoAlgoTrigRcd> EEToken_;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> FFToken_;
  edm::ESGetToken<L1GtPsbSetup, L1GtPsbSetupRcd> GGToken_;
  edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> HHToken_;
  edm::ESGetToken<L1MuDTTFMasks, L1MuDTTFMasksRcd> IIToken_;
  edm::ESGetToken<L1MuGMTChannelMask, L1MuGMTChannelMaskRcd> JJToken_;
  edm::ESGetToken<L1RCTChannelMask, L1RCTChannelMaskRcd> KKToken_;
  edm::ESGetToken<L1RCTNoisyChannelMask, L1RCTNoisyChannelMaskRcd> LLToken_;
  edm::ESGetToken<L1GctChannelMask, L1GctChannelMaskRcd> MMToken_;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> NNToken_;
  edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> OOToken_;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> PPToken_;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> QQToken_;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskVetoTechTrigRcd> RRToken_;
};

L1TConfigDumper::L1TConfigDumper(const edm::ParameterSet& iConfig)
    : AToken_(esConsumes()),
      BToken_(esConsumes()),
      CToken_(esConsumes()),
      DToken_(esConsumes()),
      EToken_(esConsumes()),
      FToken_(esConsumes()),
      GToken_(esConsumes()),
      HToken_(esConsumes()),
      IToken_(esConsumes()),
      JToken_(esConsumes()),
      KToken_(esConsumes()),
      LToken_(esConsumes()),
      MToken_(esConsumes()),
      NToken_(esConsumes()),
      OToken_(esConsumes()),
      PToken_(esConsumes()),
      QToken_(esConsumes()),
      RToken_(esConsumes()),
      SToken_(esConsumes()),
      TToken_(esConsumes()),
      UToken_(esConsumes()),
      VToken_(esConsumes()),
      WToken_(esConsumes()),
      XToken_(esConsumes()),
      YToken_(esConsumes()),
      ZToken_(esConsumes()),
      AAToken_(esConsumes()),
      BBToken_(esConsumes()),
      CCToken_(esConsumes()),
      DDToken_(esConsumes()),
      EEToken_(esConsumes()),
      FFToken_(esConsumes()),
      GGToken_(esConsumes()),
      HHToken_(esConsumes()),
      IIToken_(esConsumes()),
      JJToken_(esConsumes()),
      KKToken_(esConsumes()),
      LLToken_(esConsumes()),
      MMToken_(esConsumes()),
      NNToken_(esConsumes()),
      OOToken_(esConsumes()),
      PPToken_(esConsumes()),
      QQToken_(esConsumes()),
      RRToken_(esConsumes()) {}

void L1TConfigDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<L1TriggerKeyList> A = iSetup.getHandle(AToken_);

  edm::ESHandle<L1TriggerKey> B = iSetup.getHandle(BToken_);

  edm::ESHandle<L1CaloEtScale> C = iSetup.getHandle(CToken_);

  edm::ESHandle<L1CaloEtScale> D = iSetup.getHandle(DToken_);

  edm::ESHandle<L1CaloEtScale> E = iSetup.getHandle(EToken_);

  edm::ESHandle<L1CaloEtScale> F = iSetup.getHandle(FToken_);

  edm::ESHandle<L1MuTriggerScales> G = iSetup.getHandle(GToken_);

  edm::ESHandle<L1MuTriggerPtScale> H = iSetup.getHandle(HToken_);

  edm::ESHandle<L1MuGMTScales> I = iSetup.getHandle(IToken_);

  edm::ESHandle<L1MuCSCTFConfiguration> J = iSetup.getHandle(JToken_);

  edm::ESHandle<L1MuCSCTFAlignment> K = iSetup.getHandle(KToken_);

  edm::ESHandle<L1MuCSCPtLut> L = iSetup.getHandle(LToken_);

  edm::ESHandle<L1MuDTEtaPatternLut> M = iSetup.getHandle(MToken_);

  edm::ESHandle<L1MuDTExtLut> N = iSetup.getHandle(NToken_);

  edm::ESHandle<L1MuDTPhiLut> O = iSetup.getHandle(OToken_);

  edm::ESHandle<L1MuDTPtaLut> P = iSetup.getHandle(PToken_);

  edm::ESHandle<L1MuDTQualPatternLut> Q = iSetup.getHandle(QToken_);

  edm::ESHandle<L1MuDTTFParameters> R = iSetup.getHandle(RToken_);

  edm::ESHandle<L1RPCConfig> S = iSetup.getHandle(SToken_);

  edm::ESHandle<L1RPCConeDefinition> T = iSetup.getHandle(TToken_);

  edm::ESHandle<L1RPCHsbConfig> U = iSetup.getHandle(UToken_);

  edm::ESHandle<L1RPCBxOrConfig> V = iSetup.getHandle(VToken_);

  edm::ESHandle<L1MuGMTParameters> W = iSetup.getHandle(WToken_);

  edm::ESHandle<L1RCTParameters> X = iSetup.getHandle(XToken_);

  edm::ESHandle<L1CaloEcalScale> Y = iSetup.getHandle(YToken_);

  edm::ESHandle<L1CaloHcalScale> Z = iSetup.getHandle(ZToken_);

  edm::ESHandle<L1GctJetFinderParams> AA = iSetup.getHandle(AAToken_);

  edm::ESHandle<L1GtBoardMaps> BB = iSetup.getHandle(BBToken_);

  edm::ESHandle<L1GtParameters> CC = iSetup.getHandle(CCToken_);

  edm::ESHandle<L1GtStableParameters> DD = iSetup.getHandle(DDToken_);

  edm::ESHandle<L1GtTriggerMask> EE = iSetup.getHandle(EEToken_);

  edm::ESHandle<L1GtTriggerMenu> FF = iSetup.getHandle(FFToken_);

  edm::ESHandle<L1GtPsbSetup> GG = iSetup.getHandle(GGToken_);

  edm::ESHandle<L1CaloGeometry> HH = iSetup.getHandle(HHToken_);

  edm::ESHandle<L1MuDTTFMasks> II = iSetup.getHandle(IIToken_);

  edm::ESHandle<L1MuGMTChannelMask> JJ = iSetup.getHandle(JJToken_);

  edm::ESHandle<L1RCTChannelMask> KK = iSetup.getHandle(KKToken_);

  edm::ESHandle<L1RCTNoisyChannelMask> LL = iSetup.getHandle(LLToken_);

  edm::ESHandle<L1GctChannelMask> MM = iSetup.getHandle(MMToken_);

  edm::ESHandle<L1GtPrescaleFactors> NN = iSetup.getHandle(NNToken_);

  edm::ESHandle<L1GtPrescaleFactors> OO = iSetup.getHandle(OOToken_);

  edm::ESHandle<L1GtTriggerMask> PP = iSetup.getHandle(PPToken_);

  edm::ESHandle<L1GtTriggerMask> QQ = iSetup.getHandle(QQToken_);

  edm::ESHandle<L1GtTriggerMask> RR = iSetup.getHandle(RRToken_);

  //edm::ESHandle< NumL1Cond > SS;
  //iSetup.get< NumL1CondRcd >().get( SS) ;

  // config driven printout of payloads:
  //rctParam->print(std::cout);

  //AA->print(std::cout); // no member named 'print'
  CC->print(std::cout);
  GG->print(std::cout);
  int numberConditionChips = 1;
  FF->print(std::cout, numberConditionChips);
  J->print(std::cout);
  II->print();
  //W->print(std::cout); // no member named 'print'
  KK->print(std::cout);
  X->print(std::cout);
  //U->print(std::cout); // no member named 'print'
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TConfigDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TConfigDumper);
