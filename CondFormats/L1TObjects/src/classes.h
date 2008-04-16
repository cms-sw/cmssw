#include "CondFormats/L1TObjects/interface/BitArray.h"
#include "CondFormats/L1TObjects/interface/L1MuScale.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1MuDTExtLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPhiLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTPtaLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTEtaPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuDTQualPatternLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCPtLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCDTLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCLocalPhiLut.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCGlobalLuts.h"
#include "CondFormats/L1TObjects/interface/L1CSCTPParameters.h"
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "CondFormats/L1TObjects/interface/DTConfigBti.h"
#include "CondFormats/L1TObjects/interface/DTConfigTraco.h"
#include "CondFormats/L1TObjects/interface/DTConfigTSTheta.h"
#include "CondFormats/L1TObjects/interface/DTConfigTSPhi.h"
#include "CondFormats/L1TObjects/interface/DTConfigTrigUnit.h"
#include "CondFormats/L1TObjects/interface/DTConfigSectColl.h"
#include "CondFormats/L1TObjects/interface/DTConfigManager.h"
#include "CondFormats/L1TObjects/interface/L1RCTParameters.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/L1TObjects/interface/L1GtParameters.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/L1TObjects/interface/L1GtCondition.h"
#include "CondFormats/L1TObjects/interface/L1GtAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

namespace {
  namespace {
    std::map< std::string, std::map< std::string, std::string > > dummy0 ;
    std::map<DTBtiId,DTConfigBti> dummy1 ;
    std::map<DTTracoId,DTConfigTraco> dummy2 ;
    std::vector<L1MuDTExtLut::LUT> dummy3 ;
    std::vector<L1GtMuonTemplate> dummy4 ;
    std::vector<L1GtCaloTemplate> dummy5 ;
    std::vector<L1GtEnergySumTemplate> dummy6 ;
    std::vector<L1GtJetCountsTemplate> dummy7 ;
    std::vector<L1GtCorrelationTemplate> dummy8 ;
    std::map< std::string, L1GtAlgorithm > dummy9 ;
  }
}
