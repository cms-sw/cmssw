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

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlDefinitions.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// system include files
#include <string>

// user include files

// static class members

const std::string L1GtVhdlDefinitions::vhdlTemplateAlgoAndOr_("pre_algo_and_or.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateCondChip_("cond_chip.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateEtmSetup_("etm_setup.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateMuonSetup_("muon_setup.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateCaloSetup_("calo_setup.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateCondChipPkg1_("cond1_chip_pkg.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateCondChipPkg2_("cond2_chip_pkg.vhd");
const std::string L1GtVhdlDefinitions::vhdlTemplateDefValPkg_("def_val_pkg.vhd");
const std::string L1GtVhdlDefinitions::quartusSetupFileChip1_("cond1_chip.qsf");
const std::string L1GtVhdlDefinitions::quartusSetupFileChip2_("cond2_chip.qsf");

const std::string L1GtVhdlDefinitions::outputSubDir1_("cond1");
const std::string L1GtVhdlDefinitions::outputSubDir2_("cond2");

const std::string L1GtVhdlDefinitions::substParamAlgos_("prealgos");
const std::string L1GtVhdlDefinitions::substParamParticle_("particle");
const std::string L1GtVhdlDefinitions::substParamType_("type");
const std::string L1GtVhdlDefinitions::substParamMaxNr_("max_nr");
const std::string L1GtVhdlDefinitions::substParamDefValId_("def_val_id");
const std::string L1GtVhdlDefinitions::substParamContent_("content");
const std::string L1GtVhdlDefinitions::substParamOthers_("others");
const std::string L1GtVhdlDefinitions::substParamDefValType_("defvaltype");
const std::string L1GtVhdlDefinitions::substParamCaloOrMuon_("calo_or_muon");
const std::string L1GtVhdlDefinitions::substParamMuonDefVals_("muon_def_vals");
const std::string L1GtVhdlDefinitions::substParamCaloDefVals_("calo_def_vals");
const std::string L1GtVhdlDefinitions::substParamEsumsDefVals_("esums_def_vals");
const std::string L1GtVhdlDefinitions::substParamJetsDefVals_("jets_def_vals");
const std::string L1GtVhdlDefinitions::substParamJetCntsCommon_("jet_cnts_common");
const std::string L1GtVhdlDefinitions::substParamCharge_("charge");

const std::string L1GtVhdlDefinitions::stringConstantAlgo_("pre_algo_a");
const std::string L1GtVhdlDefinitions::stringConstantDefValId_("def_val_id");
const std::string L1GtVhdlDefinitions::stringConstantJetCountsDefVal_("jet_cnts_def_val");
const std::string L1GtVhdlDefinitions::stringConstantEsumsLowDefVal_("esums_low_def_val");
const std::string L1GtVhdlDefinitions::stringConstantEsumsLHighDefVal_("esums_high_def_val");
const std::string L1GtVhdlDefinitions::stringConstantPtLowDefVal_("ptl_def_val");
const std::string L1GtVhdlDefinitions::stringConstantPtHighDefVal_("pth_def_val");
const std::string L1GtVhdlDefinitions::stringConstantQualityDefVal_("quality_def_val");
const std::string L1GtVhdlDefinitions::stringConstantQuargeDefVal_("charge_def_val");
const std::string L1GtVhdlDefinitions::stringConstantCalo_("calo");
const std::string L1GtVhdlDefinitions::stringConstantCharge1s_("charge_1_s");
const std::string L1GtVhdlDefinitions::stringConstantCharge2s_("charge_2_s");
const std::string L1GtVhdlDefinitions::stringConstantCharge2wsc_("charge_2_wsc");
const std::string L1GtVhdlDefinitions::stringConstantCharge3s_("charge_3");
const std::string L1GtVhdlDefinitions::stringConstantCharge4s_("charge_4");
const std::string L1GtVhdlDefinitions::stringConstantCommon_("COMMON");
const std::string L1GtVhdlDefinitions::stringConstantPtl_("ptl");
const std::string L1GtVhdlDefinitions::stringConstantPth_("pth");
const std::string L1GtVhdlDefinitions::stringConstantConstantNr_("CONSTANT nr_");
const std::string L1GtVhdlDefinitions::stringConstantQuality_("quality");
const std::string L1GtVhdlDefinitions::stringConstantEsumsLow_("esums_low");
const std::string L1GtVhdlDefinitions::stringConstantEsumsHigh_("esums_high");

// constructor
L1GtVhdlDefinitions::L1GtVhdlDefinitions() {
  objType2Str_[Mu] = "muon";
  objType2Str_[NoIsoEG] = "eg";
  objType2Str_[IsoEG] = "ieg";
  objType2Str_[ForJet] = "fwdjet";
  objType2Str_[TauJet] = "tau";
  objType2Str_[CenJet] = "jet";
  objType2Str_[JetCounts] = "jet_cnts";
  objType2Str_[HTT] = "htt";
  objType2Str_[ETT] = "ett";
  objType2Str_[ETM] = "etm";

  condType2Str_[Type1s] = "1_s";
  condType2Str_[Type2s] = "2_s";
  condType2Str_[Type2wsc] = "2_wsc";
  condType2Str_[Type3s] = "3";
  condType2Str_[Type4s] = "4";
  condType2Str_[Type2cor] = "Type2cor";
  condType2Str_[TypeETM] = "cond";
  condType2Str_[TypeETT] = "cond";
  condType2Str_[TypeHTT] = "cond";
  condType2Str_[TypeJetCounts] = "jet_cnts";

  caloType2Int_[IsoEG] = "0";
  caloType2Int_[NoIsoEG] = "1";
  caloType2Int_[CenJet] = "2";
  caloType2Int_[TauJet] = "3";
  caloType2Int_[ForJet] = "4";
  caloType2Int_[Mu] = "5";
  caloType2Int_[ETM] = "6";
}

// destructor
L1GtVhdlDefinitions::~L1GtVhdlDefinitions() {
  // empty
}

const std::map<L1GtObject, std::string> L1GtVhdlDefinitions::getObj2StrMap() { return objType2Str_; }

const std::map<L1GtConditionType, std::string> L1GtVhdlDefinitions::getCond2StrMap() { return condType2Str_; }

const std::map<L1GtObject, std::string> L1GtVhdlDefinitions::getCalo2IntMap() { return caloType2Int_; }

std::string L1GtVhdlDefinitions::obj2str(const L1GtObject &type) { return objType2Str_[type]; }

std::string L1GtVhdlDefinitions::type2str(const L1GtConditionType &type) { return condType2Str_[type]; }
