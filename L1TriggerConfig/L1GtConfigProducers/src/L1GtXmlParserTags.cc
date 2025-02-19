/**
 * \class L1GtXmlParserTags
 *
 *
 * Description: Tags for the Xerces-C XML parser for the L1 Trigger menu.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 * $Date: 2009/10/29 16:37:57 $
 * $Revision: 1.9 $
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"

// system include files
#include <string>

// user include files

// constructor
L1GtXmlParserTags::L1GtXmlParserTags() {

    // empty
}

// destructor
L1GtXmlParserTags::~L1GtXmlParserTags() {

    // empty

}

// static class members

const std::string L1GtXmlParserTags::m_xmlTagDef("def");
const std::string L1GtXmlParserTags::m_xmlTagHeader("header");

const std::string L1GtXmlParserTags::m_xmlTagMenuInterface("MenuInterface");
const std::string L1GtXmlParserTags::m_xmlTagMenuInterfaceDate("MenuInterface_CreationDate");
const std::string L1GtXmlParserTags::m_xmlTagMenuInterfaceAuthor("MenuInterface_CreationAuthor");
const std::string L1GtXmlParserTags::m_xmlTagMenuInterfaceDescription("MenuInterface_Description");

const std::string L1GtXmlParserTags::m_xmlTagMenuDate("Menu_CreationDate");
const std::string L1GtXmlParserTags::m_xmlTagMenuAuthor("Menu_CreationAuthor");
const std::string L1GtXmlParserTags::m_xmlTagMenuDescription("Menu_Description");

const std::string L1GtXmlParserTags::m_xmlTagMenuAlgImpl("AlgImplementation");

const std::string L1GtXmlParserTags::m_xmlTagScaleDbKey("ScaleDbKey");


const std::string L1GtXmlParserTags::m_xmlTagChip("condition_chip_");
const std::string L1GtXmlParserTags::m_xmlTagConditions("conditions");
// see parseAlgorithms note for "prealgos"
const std::string L1GtXmlParserTags::m_xmlTagAlgorithms("prealgos");
const std::string L1GtXmlParserTags::m_xmlTagTechTriggers("techtriggers");

const std::string L1GtXmlParserTags::m_xmlAlgorithmAttrAlias("algAlias");

const std::string L1GtXmlParserTags::m_xmlConditionAttrCondition("condition");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObject("particle");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType("type");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionMuon("muon");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionCalo("calo");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionEnergySum("esums");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionJetCounts("jet_cnts");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionCastor("CondCastor");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionHfBitCounts("CondHfBitCounts");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionHfRingEtSums("CondHfRingEtSums");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionCorrelation("CondCorrelation");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionBptx("CondBptx");
const std::string L1GtXmlParserTags::m_xmlConditionAttrConditionExternal("CondExternal");

const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectMu("muon");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectNoIsoEG("eg");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectIsoEG("ieg");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectCenJet("jet");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectForJet("fwdjet");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectTauJet("tau");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectETM("etm");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectETT("ett");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectHTT("htt");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectHTM("htm");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectJetCounts("jet_cnts");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectHfBitCounts("HfBitCounts");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectHfRingEtSums("HfRingEtSums");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectCastor("Castor");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectBptx("Bptx");
const std::string L1GtXmlParserTags::m_xmlConditionAttrObjectGtExternal("GtExternal");

const std::string L1GtXmlParserTags::m_xmlConditionAttrType1s("1_s");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType2s("2_s");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType2wsc("2_wsc");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType2cor("2_cor");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType3s("3");
const std::string L1GtXmlParserTags::m_xmlConditionAttrType4s("4");
const std::string L1GtXmlParserTags::m_xmlConditionAttrTypeCastor("TypeCastor");
const std::string L1GtXmlParserTags::m_xmlConditionAttrTypeBptx("TypeBptx");
const std::string L1GtXmlParserTags::m_xmlConditionAttrTypeExternal("TypeExternal");

const std::string L1GtXmlParserTags::m_xmlAttrMode("mode");
const std::string L1GtXmlParserTags::m_xmlAttrModeBit("bit");
const std::string L1GtXmlParserTags::m_xmlAttrMax("max");

const std::string L1GtXmlParserTags::m_xmlAttrNr("nr");
const std::string L1GtXmlParserTags::m_xmlAttrPin("pin");
const std::string L1GtXmlParserTags::m_xmlAttrPinA("a");

const std::string L1GtXmlParserTags::m_xmlTagPtHighThreshold("pt_h_threshold");
const std::string L1GtXmlParserTags::m_xmlTagPtLowThreshold("pt_l_threshold");
const std::string L1GtXmlParserTags::m_xmlTagQuality("quality");
const std::string L1GtXmlParserTags::m_xmlTagEta("eta");
const std::string L1GtXmlParserTags::m_xmlTagPhi("phi");
const std::string L1GtXmlParserTags::m_xmlTagPhiHigh("phi_h");
const std::string L1GtXmlParserTags::m_xmlTagPhiLow("phi_l");
const std::string L1GtXmlParserTags::m_xmlTagChargeCorrelation("charge_correlation");
const std::string L1GtXmlParserTags::m_xmlTagEnableMip("en_mip");
const std::string L1GtXmlParserTags::m_xmlTagEnableIso("en_iso");
const std::string L1GtXmlParserTags::m_xmlTagRequestIso("request_iso");
const std::string L1GtXmlParserTags::m_xmlTagDeltaEta("delta_eta");
const std::string L1GtXmlParserTags::m_xmlTagDeltaPhi("delta_phi");

const std::string L1GtXmlParserTags::m_xmlTagEtThreshold("et_threshold");
const std::string L1GtXmlParserTags::m_xmlTagEnergyOverflow("en_overflow");

const std::string L1GtXmlParserTags::m_xmlTagCountThreshold("et_threshold");
const std::string L1GtXmlParserTags::m_xmlTagCountOverflow("en_overflow");

const std::string L1GtXmlParserTags::m_xmlTagOutput("output");
const std::string L1GtXmlParserTags::m_xmlTagOutputPin("output_pin");

const std::string L1GtXmlParserTags::m_xmlTagGEq("ge_eq");
const std::string L1GtXmlParserTags::m_xmlTagValue("value");

const std::string L1GtXmlParserTags::m_xmlTagChipDef("chip_def");
const std::string L1GtXmlParserTags::m_xmlTagChip1("chip_1");
const std::string L1GtXmlParserTags::m_xmlTagCa("ca");

// strings for the vme xml file syntax
const std::string L1GtXmlParserTags::m_xmlTagVme("vme");
const std::string L1GtXmlParserTags::m_xmlTagVmeAddress("address");
