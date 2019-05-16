#ifndef L1GtConfigProducers_L1GtVhdlDefinitions_h
#define L1GtConfigProducers_L1GtVhdlDefinitions_h

/**
 * \class L1GtVhdlDefinitions
 * 
 * 
 * Description: Contains conversion maps for conversion of trigger objects to strings etc.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Philipp Wagner
 * 
 *
 */

// system include files
#include <string>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations

// class declaration
class L1GtVhdlDefinitions {
public:
  enum VmeRegister { RegPtHighThreshold, RegPtLowThreshold, RegQualityRange, RegChargeCorrelation, RegEtThreshold };

  /// constructor
  L1GtVhdlDefinitions();

  /// destructor
  virtual ~L1GtVhdlDefinitions();

  /// converts object type to firmware string
  std::string obj2str(const L1GtObject &type);

  /// converts a condition type to firmware string
  std::string type2str(const L1GtConditionType &type);

  const std::map<L1GtObject, std::string> getObj2StrMap();

  const std::map<L1GtConditionType, std::string> getCond2StrMap();

  const std::map<L1GtObject, std::string> getCalo2IntMap();

protected:
  // templates

  static const std::string vhdlTemplateAlgoAndOr_;
  static const std::string vhdlTemplateCondChip_;
  static const std::string vhdlTemplateDefValPkg_;
  static const std::string vhdlTemplateEtmSetup_;
  static const std::string vhdlTemplateMuonSetup_;
  static const std::string vhdlTemplateCaloSetup_;
  static const std::string vhdlTemplateCondChipPkg1_;
  static const std::string vhdlTemplateCondChipPkg2_;
  static const std::string quartusSetupFileChip1_;
  static const std::string quartusSetupFileChip2_;

  // output subdirectories

  static const std::string outputSubDir1_;
  static const std::string outputSubDir2_;

  // internal templates

  // ...

  // substitution parameters

  static const std::string substParamAlgos_;
  static const std::string substParamParticle_;
  static const std::string substParamType_;
  static const std::string substParamMaxNr_;
  static const std::string substParamDefValId_;
  static const std::string substParamCaloOrMuon_;
  static const std::string substParamContent_;
  static const std::string substParamOthers_;
  static const std::string substParamDefValType_;
  static const std::string substParamMuonDefVals_;
  static const std::string substParamCaloDefVals_;
  static const std::string substParamEsumsDefVals_;
  static const std::string substParamJetsDefVals_;
  static const std::string substParamCharge_;
  static const std::string substParamJetCntsCommon_;

  //string constants

  static const std::string stringConstantAlgo_;
  static const std::string stringConstantDefValId_;
  static const std::string stringConstantJetCountsDefVal_;
  static const std::string stringConstantConstantNr_;
  static const std::string stringConstantEsumsLowDefVal_;
  static const std::string stringConstantEsumsLHighDefVal_;
  static const std::string stringConstantPtLowDefVal_;
  static const std::string stringConstantPtHighDefVal_;
  static const std::string stringConstantQualityDefVal_;
  static const std::string stringConstantQuargeDefVal_;
  static const std::string stringConstantCalo_;
  static const std::string stringConstantCharge1s_;
  static const std::string stringConstantCharge2s_;
  static const std::string stringConstantCharge2wsc_;
  static const std::string stringConstantCharge3s_;
  static const std::string stringConstantCharge4s_;
  static const std::string stringConstantCommon_;
  static const std::string stringConstantPtl_;
  static const std::string stringConstantPth_;
  static const std::string stringConstantEsumsLow_;
  static const std::string stringConstantEsumsHigh_;
  static const std::string stringConstantQuality_;

  // ... and so on

private:
  /// converts L1GtConditionType to firmware string
  std::map<L1GtObject, std::string> objType2Str_;

  /// converts L1GtObject to calo_nr
  std::map<L1GtConditionType, std::string> condType2Str_;

  /// converts L1GtObject to string
  std::map<L1GtObject, std::string> caloType2Int_;
};

#endif /*L1GtConfigProducers_L1GtVhdlDefinitions_h*/
