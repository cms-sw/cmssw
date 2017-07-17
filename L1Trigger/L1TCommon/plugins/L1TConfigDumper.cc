
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

#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h" //Record spelled out
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


class L1TConfigDumper : public edm::one::EDAnalyzer<>  {
   public:
      explicit L1TConfigDumper(const edm::ParameterSet&);
      ~L1TConfigDumper();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
};


L1TConfigDumper::L1TConfigDumper(const edm::ParameterSet& iConfig)

{

}


L1TConfigDumper::~L1TConfigDumper()
{
}

void
L1TConfigDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   

   edm::ESHandle< L1TriggerKeyList > A;
   iSetup.get< L1TriggerKeyListRcd >().get( A) ;
   
   edm::ESHandle< L1TriggerKey > B;
   iSetup.get< L1TriggerKeyRcd >().get( B) ;
   
   //edm::ESHandle< L1JetEtScale > C;
   edm::ESHandle< L1CaloEtScale > C;
   iSetup.get< L1JetEtScaleRcd >().get( C) ;
   
   //edm::ESHandle< L1EmEtScale > D;
   edm::ESHandle< L1CaloEtScale > D;
   iSetup.get< L1EmEtScaleRcd >().get( D) ;
  
   //edm::ESHandle< L1HtMissScale > E;
   edm::ESHandle< L1CaloEtScale > E;
   iSetup.get< L1HtMissScaleRcd >().get( E) ;
   
   //edm::ESHandle< L1HfRingEtScale > F;
   edm::ESHandle< L1CaloEtScale > F;
   iSetup.get< L1HfRingEtScaleRcd >().get( F) ;
   
   edm::ESHandle< L1MuTriggerScales > G;
   iSetup.get< L1MuTriggerScalesRcd >().get( G) ;
   
   edm::ESHandle< L1MuTriggerPtScale > H;
   iSetup.get< L1MuTriggerPtScaleRcd >().get( H) ;
   
   edm::ESHandle< L1MuGMTScales > I;
   iSetup.get< L1MuGMTScalesRcd >().get( I) ;
   
   edm::ESHandle< L1MuCSCTFConfiguration > J;
   iSetup.get< L1MuCSCTFConfigurationRcd >().get( J) ;
   
   edm::ESHandle< L1MuCSCTFAlignment > K;
   iSetup.get< L1MuCSCTFAlignmentRcd >().get( K) ;
   
   edm::ESHandle< L1MuCSCPtLut > L;
   iSetup.get< L1MuCSCPtLutRcd >().get( L) ;
   
   edm::ESHandle< L1MuDTEtaPatternLut > M;
   iSetup.get< L1MuDTEtaPatternLutRcd >().get( M) ;
   
   edm::ESHandle< L1MuDTExtLut > N;
   iSetup.get< L1MuDTExtLutRcd >().get( N) ;
   
   edm::ESHandle< L1MuDTPhiLut > O;
   iSetup.get< L1MuDTPhiLutRcd >().get( O) ;
   
   edm::ESHandle< L1MuDTPtaLut > P;
   iSetup.get< L1MuDTPtaLutRcd >().get( P) ;
   
   edm::ESHandle< L1MuDTQualPatternLut > Q;
   iSetup.get< L1MuDTQualPatternLutRcd >().get( Q) ;
   
   edm::ESHandle< L1MuDTTFParameters > R;
   iSetup.get< L1MuDTTFParametersRcd >().get( R) ;
   
   edm::ESHandle< L1RPCConfig > S;
   iSetup.get< L1RPCConfigRcd >().get( S) ;
   
   edm::ESHandle< L1RPCConeDefinition > T;
   iSetup.get< L1RPCConeDefinitionRcd >().get( T) ;
   
   edm::ESHandle< L1RPCHsbConfig > U;
   iSetup.get< L1RPCHsbConfigRcd >().get( U) ;
   
   edm::ESHandle< L1RPCBxOrConfig > V;
   iSetup.get< L1RPCBxOrConfigRcd >().get( V) ;
   
   edm::ESHandle< L1MuGMTParameters > W;
   iSetup.get< L1MuGMTParametersRcd >().get( W) ;
   
   edm::ESHandle< L1RCTParameters > X;
   iSetup.get< L1RCTParametersRcd >().get( X) ;
   
   edm::ESHandle< L1CaloEcalScale > Y;
   iSetup.get< L1CaloEcalScaleRcd >().get( Y) ;
   
   edm::ESHandle< L1CaloHcalScale > Z;
   iSetup.get< L1CaloHcalScaleRcd >().get( Z) ;
   
   edm::ESHandle< L1GctJetFinderParams > AA;
   iSetup.get< L1GctJetFinderParamsRcd >().get( AA) ;
   
   edm::ESHandle< L1GtBoardMaps > BB;
   iSetup.get< L1GtBoardMapsRcd >().get( BB) ;
   
   edm::ESHandle< L1GtParameters > CC;
   iSetup.get< L1GtParametersRcd >().get( CC) ;
   
   edm::ESHandle< L1GtStableParameters > DD;
   iSetup.get< L1GtStableParametersRcd >().get( DD) ;
   
   //edm::ESHandle< L1GtTriggerMaskVetoAlgoTrig > EE;
   edm::ESHandle< L1GtTriggerMask > EE;
   iSetup.get< L1GtTriggerMaskVetoAlgoTrigRcd >().get( EE) ;
   
   edm::ESHandle< L1GtTriggerMenu > FF;
   iSetup.get< L1GtTriggerMenuRcd >().get( FF) ;
   
   edm::ESHandle< L1GtPsbSetup > GG;
   iSetup.get< L1GtPsbSetupRcd >().get( GG) ;
   
   edm::ESHandle< L1CaloGeometry > HH;
   iSetup.get< L1CaloGeometryRecord >().get( HH) ; // Record spelled out
   
   edm::ESHandle< L1MuDTTFMasks > II;
   iSetup.get< L1MuDTTFMasksRcd >().get( II) ;
   
   edm::ESHandle< L1MuGMTChannelMask > JJ;
   iSetup.get< L1MuGMTChannelMaskRcd >().get( JJ) ;
   
   edm::ESHandle< L1RCTChannelMask > KK;
   iSetup.get< L1RCTChannelMaskRcd >().get( KK) ;
   
   edm::ESHandle< L1RCTNoisyChannelMask > LL;
   iSetup.get< L1RCTNoisyChannelMaskRcd >().get( LL) ;
   
   edm::ESHandle< L1GctChannelMask > MM;
   iSetup.get< L1GctChannelMaskRcd >().get( MM) ;
   
   //edm::ESHandle< L1GtPrescaleFactorsAlgoTrig > NN;
   edm::ESHandle< L1GtPrescaleFactors > NN;
   iSetup.get< L1GtPrescaleFactorsAlgoTrigRcd >().get( NN) ;
   
   //edm::ESHandle< L1GtPrescaleFactorsTechTrig > OO;
   edm::ESHandle< L1GtPrescaleFactors > OO;
   iSetup.get< L1GtPrescaleFactorsTechTrigRcd >().get( OO) ;
   
   //edm::ESHandle< L1GtTriggerMaskAlgoTrig > PP;
   edm::ESHandle< L1GtTriggerMask > PP;
   iSetup.get< L1GtTriggerMaskAlgoTrigRcd >().get( PP) ;
   
   //edm::ESHandle< L1GtTriggerMaskTechTrig > QQ;
   edm::ESHandle< L1GtTriggerMask > QQ;
   iSetup.get< L1GtTriggerMaskTechTrigRcd >().get( QQ) ;
   
   //edm::ESHandle< L1GtTriggerMaskVetoTechTrig > RR;
   edm::ESHandle< L1GtTriggerMask > RR;
   iSetup.get< L1GtTriggerMaskVetoTechTrigRcd >().get( RR) ;
   
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

void 
L1TConfigDumper::beginJob()
{
}

void 
L1TConfigDumper::endJob() 
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
L1TConfigDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TConfigDumper);
