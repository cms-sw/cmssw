
#include "CondFormats/L1TObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void L1CaloEcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-scale", m_scale);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloEcalScale);

template <class Archive>
void L1CaloEtScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-linScaleMax", m_linScaleMax);
    ar & boost::serialization::make_nvp("m-rankScaleMax", m_rankScaleMax);
    ar & boost::serialization::make_nvp("m-linearLsb", m_linearLsb);
    ar & boost::serialization::make_nvp("m-thresholds", m_thresholds);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloEtScale);

template <class Archive>
void L1CaloGeometry::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-version", m_version);
    ar & boost::serialization::make_nvp("m-numberGctEmJetPhiBins", m_numberGctEmJetPhiBins);
    ar & boost::serialization::make_nvp("m-numberGctEtSumPhiBins", m_numberGctEtSumPhiBins);
    ar & boost::serialization::make_nvp("m-numberGctHtSumPhiBins", m_numberGctHtSumPhiBins);
    ar & boost::serialization::make_nvp("m-numberGctCentralEtaBinsPerHalf", m_numberGctCentralEtaBinsPerHalf);
    ar & boost::serialization::make_nvp("m-numberGctForwardEtaBinsPerHalf", m_numberGctForwardEtaBinsPerHalf);
    ar & boost::serialization::make_nvp("m-etaSignBitOffset", m_etaSignBitOffset);
    ar & boost::serialization::make_nvp("m-gctEtaBinBoundaries", m_gctEtaBinBoundaries);
    ar & boost::serialization::make_nvp("m-etaBinsPerHalf", m_etaBinsPerHalf);
    ar & boost::serialization::make_nvp("m-gctEmJetPhiBinWidth", m_gctEmJetPhiBinWidth);
    ar & boost::serialization::make_nvp("m-gctEtSumPhiBinWidth", m_gctEtSumPhiBinWidth);
    ar & boost::serialization::make_nvp("m-gctHtSumPhiBinWidth", m_gctHtSumPhiBinWidth);
    ar & boost::serialization::make_nvp("m-gctEmJetPhiOffset", m_gctEmJetPhiOffset);
    ar & boost::serialization::make_nvp("m-gctEtSumPhiOffset", m_gctEtSumPhiOffset);
    ar & boost::serialization::make_nvp("m-gctHtSumPhiOffset", m_gctHtSumPhiOffset);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloGeometry);

template <class Archive>
void L1CaloHcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-scale", m_scale);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloHcalScale);

template <class Archive>
void L1GctChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("emCrateMask-", emCrateMask_);
    ar & boost::serialization::make_nvp("regionMask-", regionMask_);
    ar & boost::serialization::make_nvp("tetMask-", tetMask_);
    ar & boost::serialization::make_nvp("metMask-", metMask_);
    ar & boost::serialization::make_nvp("htMask-", htMask_);
    ar & boost::serialization::make_nvp("mhtMask-", mhtMask_);
}
COND_SERIALIZATION_INSTANTIATE(L1GctChannelMask);

template <class Archive>
void L1GctJetFinderParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("rgnEtLsb-", rgnEtLsb_);
    ar & boost::serialization::make_nvp("htLsb-", htLsb_);
    ar & boost::serialization::make_nvp("cenJetEtSeed-", cenJetEtSeed_);
    ar & boost::serialization::make_nvp("forJetEtSeed-", forJetEtSeed_);
    ar & boost::serialization::make_nvp("tauJetEtSeed-", tauJetEtSeed_);
    ar & boost::serialization::make_nvp("tauIsoEtThreshold-", tauIsoEtThreshold_);
    ar & boost::serialization::make_nvp("htJetEtThreshold-", htJetEtThreshold_);
    ar & boost::serialization::make_nvp("mhtJetEtThreshold-", mhtJetEtThreshold_);
    ar & boost::serialization::make_nvp("cenForJetEtaBoundary-", cenForJetEtaBoundary_);
    ar & boost::serialization::make_nvp("corrType-", corrType_);
    ar & boost::serialization::make_nvp("jetCorrCoeffs-", jetCorrCoeffs_);
    ar & boost::serialization::make_nvp("tauCorrCoeffs-", tauCorrCoeffs_);
    ar & boost::serialization::make_nvp("convertToEnergy-", convertToEnergy_);
    ar & boost::serialization::make_nvp("energyConversionCoeffs-", energyConversionCoeffs_);
}
COND_SERIALIZATION_INSTANTIATE(L1GctJetFinderParams);

template <class Archive>
void L1GtAlgorithm::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-algoName", m_algoName);
    ar & boost::serialization::make_nvp("m-algoAlias", m_algoAlias);
    ar & boost::serialization::make_nvp("m-algoLogicalExpression", m_algoLogicalExpression);
    ar & boost::serialization::make_nvp("m-algoRpnVector", m_algoRpnVector);
    ar & boost::serialization::make_nvp("m-algoBitNumber", m_algoBitNumber);
    ar & boost::serialization::make_nvp("m-algoChipNumber", m_algoChipNumber);
}
COND_SERIALIZATION_INSTANTIATE(L1GtAlgorithm);

template <class Archive>
void L1GtBoard::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-gtBoardType", m_gtBoardType);
    ar & boost::serialization::make_nvp("m-gtBoardIndex", m_gtBoardIndex);
    ar & boost::serialization::make_nvp("m-gtPositionDaqRecord", m_gtPositionDaqRecord);
    ar & boost::serialization::make_nvp("m-gtPositionEvmRecord", m_gtPositionEvmRecord);
    ar & boost::serialization::make_nvp("m-gtBitDaqActiveBoards", m_gtBitDaqActiveBoards);
    ar & boost::serialization::make_nvp("m-gtBitEvmActiveBoards", m_gtBitEvmActiveBoards);
    ar & boost::serialization::make_nvp("m-gtBoardSlot", m_gtBoardSlot);
    ar & boost::serialization::make_nvp("m-gtBoardHexName", m_gtBoardHexName);
    ar & boost::serialization::make_nvp("m-gtQuadInPsb", m_gtQuadInPsb);
    ar & boost::serialization::make_nvp("m-gtInputPsbChannels", m_gtInputPsbChannels);
}
COND_SERIALIZATION_INSTANTIATE(L1GtBoard);

template <class Archive>
void L1GtBoardMaps::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-gtBoardMaps", m_gtBoardMaps);
}
COND_SERIALIZATION_INSTANTIATE(L1GtBoardMaps);

template <class Archive>
void L1GtBptxTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}
COND_SERIALIZATION_INSTANTIATE(L1GtBptxTemplate);

template <class Archive>
void L1GtCaloTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
    ar & boost::serialization::make_nvp("m-correlationParameter", m_correlationParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCaloTemplate);

template <class Archive>
void L1GtCaloTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("deltaEtaRange", deltaEtaRange);
    ar & boost::serialization::make_nvp("deltaPhiRange", deltaPhiRange);
    ar & boost::serialization::make_nvp("deltaPhiMaxbits", deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCaloTemplate::CorrelationParameter);

template <class Archive>
void L1GtCaloTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("etThreshold", etThreshold);
    ar & boost::serialization::make_nvp("etaRange", etaRange);
    ar & boost::serialization::make_nvp("phiRange", phiRange);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCaloTemplate::ObjectParameter);

template <class Archive>
void L1GtCastorTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}
COND_SERIALIZATION_INSTANTIATE(L1GtCastorTemplate);

template <class Archive>
void L1GtCondition::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-condName", m_condName);
    ar & boost::serialization::make_nvp("m-condCategory", m_condCategory);
    ar & boost::serialization::make_nvp("m-condType", m_condType);
    ar & boost::serialization::make_nvp("m-objectType", m_objectType);
    ar & boost::serialization::make_nvp("m-condGEq", m_condGEq);
    ar & boost::serialization::make_nvp("m-condChipNr", m_condChipNr);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCondition);

template <class Archive>
void L1GtCorrelationTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-cond0Category", m_cond0Category);
    ar & boost::serialization::make_nvp("m-cond1Category", m_cond1Category);
    ar & boost::serialization::make_nvp("m-cond0Index", m_cond0Index);
    ar & boost::serialization::make_nvp("m-cond1Index", m_cond1Index);
    ar & boost::serialization::make_nvp("m-correlationParameter", m_correlationParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCorrelationTemplate);

template <class Archive>
void L1GtCorrelationTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("deltaEtaRange", deltaEtaRange);
    ar & boost::serialization::make_nvp("deltaPhiRange", deltaPhiRange);
    ar & boost::serialization::make_nvp("deltaPhiMaxbits", deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCorrelationTemplate::CorrelationParameter);

template <class Archive>
void L1GtEnergySumTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtEnergySumTemplate);

template <class Archive>
void L1GtEnergySumTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("etThreshold", etThreshold);
    ar & boost::serialization::make_nvp("energyOverflow", energyOverflow);
    ar & boost::serialization::make_nvp("phiRange0Word", phiRange0Word);
    ar & boost::serialization::make_nvp("phiRange1Word", phiRange1Word);
}
COND_SERIALIZATION_INSTANTIATE(L1GtEnergySumTemplate::ObjectParameter);

template <class Archive>
void L1GtExternalTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}
COND_SERIALIZATION_INSTANTIATE(L1GtExternalTemplate);

template <class Archive>
void L1GtHfBitCountsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfBitCountsTemplate);

template <class Archive>
void L1GtHfBitCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("countIndex", countIndex);
    ar & boost::serialization::make_nvp("countThreshold", countThreshold);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfBitCountsTemplate::ObjectParameter);

template <class Archive>
void L1GtHfRingEtSumsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfRingEtSumsTemplate);

template <class Archive>
void L1GtHfRingEtSumsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("etSumIndex", etSumIndex);
    ar & boost::serialization::make_nvp("etSumThreshold", etSumThreshold);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfRingEtSumsTemplate::ObjectParameter);

template <class Archive>
void L1GtJetCountsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtJetCountsTemplate);

template <class Archive>
void L1GtJetCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("countIndex", countIndex);
    ar & boost::serialization::make_nvp("countThreshold", countThreshold);
    ar & boost::serialization::make_nvp("countOverflow", countOverflow);
}
COND_SERIALIZATION_INSTANTIATE(L1GtJetCountsTemplate::ObjectParameter);

template <class Archive>
void L1GtMuonTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & boost::serialization::make_nvp("m-objectParameter", m_objectParameter);
    ar & boost::serialization::make_nvp("m-correlationParameter", m_correlationParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate);

template <class Archive>
void L1GtMuonTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("chargeCorrelation", chargeCorrelation);
    ar & boost::serialization::make_nvp("deltaEtaRange", deltaEtaRange);
    ar & boost::serialization::make_nvp("deltaPhiRange0Word", deltaPhiRange0Word);
    ar & boost::serialization::make_nvp("deltaPhiRange1Word", deltaPhiRange1Word);
    ar & boost::serialization::make_nvp("deltaPhiMaxbits", deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate::CorrelationParameter);

template <class Archive>
void L1GtMuonTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ptHighThreshold", ptHighThreshold);
    ar & boost::serialization::make_nvp("ptLowThreshold", ptLowThreshold);
    ar & boost::serialization::make_nvp("enableMip", enableMip);
    ar & boost::serialization::make_nvp("enableIso", enableIso);
    ar & boost::serialization::make_nvp("requestIso", requestIso);
    ar & boost::serialization::make_nvp("qualityRange", qualityRange);
    ar & boost::serialization::make_nvp("etaRange", etaRange);
    ar & boost::serialization::make_nvp("phiHigh", phiHigh);
    ar & boost::serialization::make_nvp("phiLow", phiLow);
}
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate::ObjectParameter);

template <class Archive>
void L1GtParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-totalBxInEvent", m_totalBxInEvent);
    ar & boost::serialization::make_nvp("m-daqActiveBoards", m_daqActiveBoards);
    ar & boost::serialization::make_nvp("m-evmActiveBoards", m_evmActiveBoards);
    ar & boost::serialization::make_nvp("m-daqNrBxBoard", m_daqNrBxBoard);
    ar & boost::serialization::make_nvp("m-evmNrBxBoard", m_evmNrBxBoard);
    ar & boost::serialization::make_nvp("m-bstLengthBytes", m_bstLengthBytes);
}
COND_SERIALIZATION_INSTANTIATE(L1GtParameters);

template <class Archive>
void L1GtPrescaleFactors::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-prescaleFactors", m_prescaleFactors);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPrescaleFactors);

template <class Archive>
void L1GtPsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-gtBoardSlot", m_gtBoardSlot);
    ar & boost::serialization::make_nvp("m-gtPsbCh0SendLvds", m_gtPsbCh0SendLvds);
    ar & boost::serialization::make_nvp("m-gtPsbCh1SendLvds", m_gtPsbCh1SendLvds);
    ar & boost::serialization::make_nvp("m-gtPsbEnableRecLvds", m_gtPsbEnableRecLvds);
    ar & boost::serialization::make_nvp("m-gtPsbEnableRecSerLink", m_gtPsbEnableRecSerLink);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPsbConfig);

template <class Archive>
void L1GtPsbSetup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-gtPsbSetup", m_gtPsbSetup);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPsbSetup);

template <class Archive>
void L1GtStableParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-numberPhysTriggers", m_numberPhysTriggers);
    ar & boost::serialization::make_nvp("m-numberPhysTriggersExtended", m_numberPhysTriggersExtended);
    ar & boost::serialization::make_nvp("m-numberTechnicalTriggers", m_numberTechnicalTriggers);
    ar & boost::serialization::make_nvp("m-numberL1Mu", m_numberL1Mu);
    ar & boost::serialization::make_nvp("m-numberL1NoIsoEG", m_numberL1NoIsoEG);
    ar & boost::serialization::make_nvp("m-numberL1IsoEG", m_numberL1IsoEG);
    ar & boost::serialization::make_nvp("m-numberL1CenJet", m_numberL1CenJet);
    ar & boost::serialization::make_nvp("m-numberL1ForJet", m_numberL1ForJet);
    ar & boost::serialization::make_nvp("m-numberL1TauJet", m_numberL1TauJet);
    ar & boost::serialization::make_nvp("m-numberL1JetCounts", m_numberL1JetCounts);
    ar & boost::serialization::make_nvp("m-numberConditionChips", m_numberConditionChips);
    ar & boost::serialization::make_nvp("m-pinsOnConditionChip", m_pinsOnConditionChip);
    ar & boost::serialization::make_nvp("m-orderConditionChip", m_orderConditionChip);
    ar & boost::serialization::make_nvp("m-numberPsbBoards", m_numberPsbBoards);
    ar & boost::serialization::make_nvp("m-ifCaloEtaNumberBits", m_ifCaloEtaNumberBits);
    ar & boost::serialization::make_nvp("m-ifMuEtaNumberBits", m_ifMuEtaNumberBits);
    ar & boost::serialization::make_nvp("m-wordLength", m_wordLength);
    ar & boost::serialization::make_nvp("m-unitLength", m_unitLength);
}
COND_SERIALIZATION_INSTANTIATE(L1GtStableParameters);

template <class Archive>
void L1GtTriggerMask::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-triggerMask", m_triggerMask);
}
COND_SERIALIZATION_INSTANTIATE(L1GtTriggerMask);

template <class Archive>
void L1GtTriggerMenu::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-triggerMenuInterface", m_triggerMenuInterface);
    ar & boost::serialization::make_nvp("m-triggerMenuName", m_triggerMenuName);
    ar & boost::serialization::make_nvp("m-triggerMenuImplementation", m_triggerMenuImplementation);
    ar & boost::serialization::make_nvp("m-scaleDbKey", m_scaleDbKey);
    ar & boost::serialization::make_nvp("m-vecMuonTemplate", m_vecMuonTemplate);
    ar & boost::serialization::make_nvp("m-vecCaloTemplate", m_vecCaloTemplate);
    ar & boost::serialization::make_nvp("m-vecEnergySumTemplate", m_vecEnergySumTemplate);
    ar & boost::serialization::make_nvp("m-vecJetCountsTemplate", m_vecJetCountsTemplate);
    ar & boost::serialization::make_nvp("m-vecCastorTemplate", m_vecCastorTemplate);
    ar & boost::serialization::make_nvp("m-vecHfBitCountsTemplate", m_vecHfBitCountsTemplate);
    ar & boost::serialization::make_nvp("m-vecHfRingEtSumsTemplate", m_vecHfRingEtSumsTemplate);
    ar & boost::serialization::make_nvp("m-vecBptxTemplate", m_vecBptxTemplate);
    ar & boost::serialization::make_nvp("m-vecExternalTemplate", m_vecExternalTemplate);
    ar & boost::serialization::make_nvp("m-vecCorrelationTemplate", m_vecCorrelationTemplate);
    ar & boost::serialization::make_nvp("m-corMuonTemplate", m_corMuonTemplate);
    ar & boost::serialization::make_nvp("m-corCaloTemplate", m_corCaloTemplate);
    ar & boost::serialization::make_nvp("m-corEnergySumTemplate", m_corEnergySumTemplate);
    ar & boost::serialization::make_nvp("m-algorithmMap", m_algorithmMap);
    ar & boost::serialization::make_nvp("m-algorithmAliasMap", m_algorithmAliasMap);
    ar & boost::serialization::make_nvp("m-technicalTriggerMap", m_technicalTriggerMap);
}
COND_SERIALIZATION_INSTANTIATE(L1GtTriggerMenu);

template <class Archive>
void L1MuBinnedScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuScale", boost::serialization::base_object<L1MuScale>(*this));
    ar & boost::serialization::make_nvp("m-nbits", m_nbits);
    ar & boost::serialization::make_nvp("m-signedPacking", m_signedPacking);
    ar & boost::serialization::make_nvp("m-NBins", m_NBins);
    ar & boost::serialization::make_nvp("m-idxoffset", m_idxoffset);
    ar & boost::serialization::make_nvp("m-Scale", m_Scale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuBinnedScale);

template <class Archive>
void L1MuCSCPtLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("pt-lut", pt_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCPtLut);

template <class Archive>
void L1MuCSCTFAlignment::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("coefficients", coefficients);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCTFAlignment);

template <class Archive>
void L1MuCSCTFConfiguration::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("registers", registers);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCTFConfiguration);

template <class Archive>
void L1MuDTEtaPattern::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-id", m_id);
    ar & boost::serialization::make_nvp("m-wheel", m_wheel);
    ar & boost::serialization::make_nvp("m-position", m_position);
    ar & boost::serialization::make_nvp("m-eta", m_eta);
    ar & boost::serialization::make_nvp("m-qual", m_qual);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTEtaPattern);

template <class Archive>
void L1MuDTEtaPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-lut", m_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTEtaPatternLut);

template <class Archive>
void L1MuDTExtLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ext-lut", ext_lut);
    ar & boost::serialization::make_nvp("nbit-phi", nbit_phi);
    ar & boost::serialization::make_nvp("nbit-phib", nbit_phib);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTExtLut);

template <class Archive>
void L1MuDTExtLut::LUT::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("low", low);
    ar & boost::serialization::make_nvp("high", high);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTExtLut::LUT);

template <class Archive>
void L1MuDTPhiLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("phi-lut", phi_lut);
    ar & boost::serialization::make_nvp("nbit-phi", nbit_phi);
    ar & boost::serialization::make_nvp("nbit-phib", nbit_phib);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTPhiLut);

template <class Archive>
void L1MuDTPtaLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("pta-lut", pta_lut);
    ar & boost::serialization::make_nvp("pta-threshold", pta_threshold);
    ar & boost::serialization::make_nvp("nbit-phi", nbit_phi);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTPtaLut);

template <class Archive>
void L1MuDTQualPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-lut", m_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTQualPatternLut);

template <class Archive>
void L1MuDTTFMasks::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("inrec-chdis-st1", inrec_chdis_st1);
    ar & boost::serialization::make_nvp("inrec-chdis-st2", inrec_chdis_st2);
    ar & boost::serialization::make_nvp("inrec-chdis-st3", inrec_chdis_st3);
    ar & boost::serialization::make_nvp("inrec-chdis-st4", inrec_chdis_st4);
    ar & boost::serialization::make_nvp("inrec-chdis-csc", inrec_chdis_csc);
    ar & boost::serialization::make_nvp("etsoc-chdis-st1", etsoc_chdis_st1);
    ar & boost::serialization::make_nvp("etsoc-chdis-st2", etsoc_chdis_st2);
    ar & boost::serialization::make_nvp("etsoc-chdis-st3", etsoc_chdis_st3);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTTFMasks);

template <class Archive>
void L1MuDTTFParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("inrec-qual-st1", inrec_qual_st1);
    ar & boost::serialization::make_nvp("inrec-qual-st2", inrec_qual_st2);
    ar & boost::serialization::make_nvp("inrec-qual-st3", inrec_qual_st3);
    ar & boost::serialization::make_nvp("inrec-qual-st4", inrec_qual_st4);
    ar & boost::serialization::make_nvp("soc-stdis-n", soc_stdis_n);
    ar & boost::serialization::make_nvp("soc-stdis-wl", soc_stdis_wl);
    ar & boost::serialization::make_nvp("soc-stdis-wr", soc_stdis_wr);
    ar & boost::serialization::make_nvp("soc-stdis-zl", soc_stdis_zl);
    ar & boost::serialization::make_nvp("soc-stdis-zr", soc_stdis_zr);
    ar & boost::serialization::make_nvp("soc-qcut-st1", soc_qcut_st1);
    ar & boost::serialization::make_nvp("soc-qcut-st2", soc_qcut_st2);
    ar & boost::serialization::make_nvp("soc-qcut-st4", soc_qcut_st4);
    ar & boost::serialization::make_nvp("soc-qual-csc", soc_qual_csc);
    ar & boost::serialization::make_nvp("soc-run-21", soc_run_21);
    ar & boost::serialization::make_nvp("soc-nbx-del", soc_nbx_del);
    ar & boost::serialization::make_nvp("soc-csc-etacanc", soc_csc_etacanc);
    ar & boost::serialization::make_nvp("soc-openlut-extr", soc_openlut_extr);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTTFParameters);

template <class Archive>
void L1MuGMTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-SubsystemMask", m_SubsystemMask);
}
COND_SERIALIZATION_INSTANTIATE(L1MuGMTChannelMask);

template <class Archive>
void L1MuGMTParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-EtaWeight-barrel", m_EtaWeight_barrel);
    ar & boost::serialization::make_nvp("m-PhiWeight-barrel", m_PhiWeight_barrel);
    ar & boost::serialization::make_nvp("m-EtaPhiThreshold-barrel", m_EtaPhiThreshold_barrel);
    ar & boost::serialization::make_nvp("m-EtaWeight-endcap", m_EtaWeight_endcap);
    ar & boost::serialization::make_nvp("m-PhiWeight-endcap", m_PhiWeight_endcap);
    ar & boost::serialization::make_nvp("m-EtaPhiThreshold-endcap", m_EtaPhiThreshold_endcap);
    ar & boost::serialization::make_nvp("m-EtaWeight-COU", m_EtaWeight_COU);
    ar & boost::serialization::make_nvp("m-PhiWeight-COU", m_PhiWeight_COU);
    ar & boost::serialization::make_nvp("m-EtaPhiThreshold-COU", m_EtaPhiThreshold_COU);
    ar & boost::serialization::make_nvp("m-CaloTrigger", m_CaloTrigger);
    ar & boost::serialization::make_nvp("m-IsolationCellSizeEta", m_IsolationCellSizeEta);
    ar & boost::serialization::make_nvp("m-IsolationCellSizePhi", m_IsolationCellSizePhi);
    ar & boost::serialization::make_nvp("m-DoOvlRpcAnd", m_DoOvlRpcAnd);
    ar & boost::serialization::make_nvp("m-PropagatePhi", m_PropagatePhi);
    ar & boost::serialization::make_nvp("m-MergeMethodPhiBrl", m_MergeMethodPhiBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodPhiFwd", m_MergeMethodPhiFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodEtaBrl", m_MergeMethodEtaBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodEtaFwd", m_MergeMethodEtaFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodPtBrl", m_MergeMethodPtBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodPtFwd", m_MergeMethodPtFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodChargeBrl", m_MergeMethodChargeBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodChargeFwd", m_MergeMethodChargeFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodMIPBrl", m_MergeMethodMIPBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodMIPFwd", m_MergeMethodMIPFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodMIPSpecialUseANDBrl", m_MergeMethodMIPSpecialUseANDBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodMIPSpecialUseANDFwd", m_MergeMethodMIPSpecialUseANDFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodISOBrl", m_MergeMethodISOBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodISOFwd", m_MergeMethodISOFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodISOSpecialUseANDBrl", m_MergeMethodISOSpecialUseANDBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodISOSpecialUseANDFwd", m_MergeMethodISOSpecialUseANDFwd);
    ar & boost::serialization::make_nvp("m-MergeMethodSRKBrl", m_MergeMethodSRKBrl);
    ar & boost::serialization::make_nvp("m-MergeMethodSRKFwd", m_MergeMethodSRKFwd);
    ar & boost::serialization::make_nvp("m-HaloOverwritesMatchedBrl", m_HaloOverwritesMatchedBrl);
    ar & boost::serialization::make_nvp("m-HaloOverwritesMatchedFwd", m_HaloOverwritesMatchedFwd);
    ar & boost::serialization::make_nvp("m-SortRankOffsetBrl", m_SortRankOffsetBrl);
    ar & boost::serialization::make_nvp("m-SortRankOffsetFwd", m_SortRankOffsetFwd);
    ar & boost::serialization::make_nvp("m-CDLConfigWordDTCSC", m_CDLConfigWordDTCSC);
    ar & boost::serialization::make_nvp("m-CDLConfigWordCSCDT", m_CDLConfigWordCSCDT);
    ar & boost::serialization::make_nvp("m-CDLConfigWordbRPCCSC", m_CDLConfigWordbRPCCSC);
    ar & boost::serialization::make_nvp("m-CDLConfigWordfRPCDT", m_CDLConfigWordfRPCDT);
    ar & boost::serialization::make_nvp("m-VersionSortRankEtaQLUT", m_VersionSortRankEtaQLUT);
    ar & boost::serialization::make_nvp("m-VersionLUTs", m_VersionLUTs);
}
COND_SERIALIZATION_INSTANTIATE(L1MuGMTParameters);

template <class Archive>
void L1MuGMTScales::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-ReducedEtaScale", m_ReducedEtaScale);
    ar & boost::serialization::make_nvp("m-DeltaEtaScale", m_DeltaEtaScale);
    ar & boost::serialization::make_nvp("m-DeltaPhiScale", m_DeltaPhiScale);
    ar & boost::serialization::make_nvp("m-OvlEtaScale", m_OvlEtaScale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuGMTScales);

template <class Archive>
void L1MuPacking::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(L1MuPacking);

template <class Archive>
void L1MuPseudoSignedPacking::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuPacking", boost::serialization::base_object<L1MuPacking>(*this));
    ar & boost::serialization::make_nvp("m-nbits", m_nbits);
}
COND_SERIALIZATION_INSTANTIATE(L1MuPseudoSignedPacking);

template <class Archive>
void L1MuScale::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(L1MuScale);

template <class Archive>
void L1MuSymmetricBinnedScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuScale", boost::serialization::base_object<L1MuScale>(*this));
    ar & boost::serialization::make_nvp("m-packing", m_packing);
    ar & boost::serialization::make_nvp("m-NBins", m_NBins);
    ar & boost::serialization::make_nvp("m-Scale", m_Scale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuSymmetricBinnedScale);

template <class Archive>
void L1MuTriggerPtScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-PtScale", m_PtScale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuTriggerPtScale);

template <class Archive>
void L1MuTriggerScales::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-RegionalEtaScale", m_RegionalEtaScale);
    ar & boost::serialization::make_nvp("m-RegionalEtaScaleCSC", m_RegionalEtaScaleCSC);
    ar & boost::serialization::make_nvp("m-GMTEtaScale", m_GMTEtaScale);
    ar & boost::serialization::make_nvp("m-PhiScale", m_PhiScale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuTriggerScales);

template <class Archive>
void L1RCTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ecalMask", ecalMask);
    ar & boost::serialization::make_nvp("hcalMask", hcalMask);
    ar & boost::serialization::make_nvp("hfMask", hfMask);
}
COND_SERIALIZATION_INSTANTIATE(L1RCTChannelMask);

template <class Archive>
void L1RCTNoisyChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ecalMask", ecalMask);
    ar & boost::serialization::make_nvp("hcalMask", hcalMask);
    ar & boost::serialization::make_nvp("hfMask", hfMask);
    ar & boost::serialization::make_nvp("ecalThreshold", ecalThreshold);
    ar & boost::serialization::make_nvp("hcalThreshold", hcalThreshold);
    ar & boost::serialization::make_nvp("hfThreshold", hfThreshold);
}
COND_SERIALIZATION_INSTANTIATE(L1RCTNoisyChannelMask);

template <class Archive>
void L1RCTParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("eGammaLSB-", eGammaLSB_);
    ar & boost::serialization::make_nvp("jetMETLSB-", jetMETLSB_);
    ar & boost::serialization::make_nvp("eMinForFGCut-", eMinForFGCut_);
    ar & boost::serialization::make_nvp("eMaxForFGCut-", eMaxForFGCut_);
    ar & boost::serialization::make_nvp("hOeCut-", hOeCut_);
    ar & boost::serialization::make_nvp("eMinForHoECut-", eMinForHoECut_);
    ar & boost::serialization::make_nvp("eMaxForHoECut-", eMaxForHoECut_);
    ar & boost::serialization::make_nvp("hMinForHoECut-", hMinForHoECut_);
    ar & boost::serialization::make_nvp("eActivityCut-", eActivityCut_);
    ar & boost::serialization::make_nvp("hActivityCut-", hActivityCut_);
    ar & boost::serialization::make_nvp("eicIsolationThreshold-", eicIsolationThreshold_);
    ar & boost::serialization::make_nvp("jscQuietThresholdBarrel-", jscQuietThresholdBarrel_);
    ar & boost::serialization::make_nvp("jscQuietThresholdEndcap-", jscQuietThresholdEndcap_);
    ar & boost::serialization::make_nvp("noiseVetoHB-", noiseVetoHB_);
    ar & boost::serialization::make_nvp("noiseVetoHEplus-", noiseVetoHEplus_);
    ar & boost::serialization::make_nvp("noiseVetoHEminus-", noiseVetoHEminus_);
    ar & boost::serialization::make_nvp("useCorrections-", useCorrections_);
    ar & boost::serialization::make_nvp("eGammaECalScaleFactors-", eGammaECalScaleFactors_);
    ar & boost::serialization::make_nvp("eGammaHCalScaleFactors-", eGammaHCalScaleFactors_);
    ar & boost::serialization::make_nvp("jetMETECalScaleFactors-", jetMETECalScaleFactors_);
    ar & boost::serialization::make_nvp("jetMETHCalScaleFactors-", jetMETHCalScaleFactors_);
    ar & boost::serialization::make_nvp("ecal-calib-", ecal_calib_);
    ar & boost::serialization::make_nvp("hcal-calib-", hcal_calib_);
    ar & boost::serialization::make_nvp("hcal-high-calib-", hcal_high_calib_);
    ar & boost::serialization::make_nvp("cross-terms-", cross_terms_);
    ar & boost::serialization::make_nvp("HoverE-smear-low-", HoverE_smear_low_);
    ar & boost::serialization::make_nvp("HoverE-smear-high-", HoverE_smear_high_);
}
COND_SERIALIZATION_INSTANTIATE(L1RCTParameters);

template <class Archive>
void L1RPCBxOrConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-firstBX", m_firstBX);
    ar & boost::serialization::make_nvp("m-lastBX", m_lastBX);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCBxOrConfig);

template <class Archive>
void L1RPCConeDefinition::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-firstTower", m_firstTower);
    ar & boost::serialization::make_nvp("m-lastTower", m_lastTower);
    ar & boost::serialization::make_nvp("m-LPSizeVec", m_LPSizeVec);
    ar & boost::serialization::make_nvp("m-ringToTowerVec", m_ringToTowerVec);
    ar & boost::serialization::make_nvp("m-ringToLPVec", m_ringToLPVec);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition);

template <class Archive>
void L1RPCConeDefinition::TLPSize::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tower", m_tower);
    ar & boost::serialization::make_nvp("m-LP", m_LP);
    ar & boost::serialization::make_nvp("m-size", m_size);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TLPSize);

template <class Archive>
void L1RPCConeDefinition::TRingToLP::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-etaPart", m_etaPart);
    ar & boost::serialization::make_nvp("m-hwPlane", m_hwPlane);
    ar & boost::serialization::make_nvp("m-LP", m_LP);
    ar & boost::serialization::make_nvp("m-index", m_index);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TRingToLP);

template <class Archive>
void L1RPCConeDefinition::TRingToTower::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-etaPart", m_etaPart);
    ar & boost::serialization::make_nvp("m-hwPlane", m_hwPlane);
    ar & boost::serialization::make_nvp("m-tower", m_tower);
    ar & boost::serialization::make_nvp("m-index", m_index);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TRingToTower);

template <class Archive>
void L1RPCConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-pats", m_pats);
    ar & boost::serialization::make_nvp("m-quals", m_quals);
    ar & boost::serialization::make_nvp("m-ppt", m_ppt);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConfig);

template <class Archive>
void L1RPCHsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-hsb0", m_hsb0);
    ar & boost::serialization::make_nvp("m-hsb1", m_hsb1);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCHsbConfig);

template <class Archive>
void L1TGlobalParameters::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-totalBxInEvent", m_totalBxInEvent);
    ar & boost::serialization::make_nvp("m-numberPhysTriggers", m_numberPhysTriggers);
    ar & boost::serialization::make_nvp("m-numberL1Mu", m_numberL1Mu);
    ar & boost::serialization::make_nvp("m-numberL1EG", m_numberL1EG);
    ar & boost::serialization::make_nvp("m-numberL1Jet", m_numberL1Jet);
    ar & boost::serialization::make_nvp("m-numberL1Tau", m_numberL1Tau);
    ar & boost::serialization::make_nvp("m-numberChips", m_numberChips);
    ar & boost::serialization::make_nvp("m-pinsOnChip", m_pinsOnChip);
    ar & boost::serialization::make_nvp("m-orderOfChip", m_orderOfChip);
    ar & boost::serialization::make_nvp("m-version", m_version);
    ar & boost::serialization::make_nvp("m-exp-ints", m_exp_ints);
    ar & boost::serialization::make_nvp("m-exp-doubles", m_exp_doubles);
}
COND_SERIALIZATION_INSTANTIATE(L1TGlobalParameters);

template <class Archive>
void L1TGlobalPrescalesVetos::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("prescale-table-", prescale_table_);
    ar & boost::serialization::make_nvp("bxmask-default-", bxmask_default_);
    ar & boost::serialization::make_nvp("bxmask-map-", bxmask_map_);
    ar & boost::serialization::make_nvp("veto-", veto_);
    ar & boost::serialization::make_nvp("exp-ints-", exp_ints_);
    ar & boost::serialization::make_nvp("exp-doubles-", exp_doubles_);
}
COND_SERIALIZATION_INSTANTIATE(L1TGlobalPrescalesVetos);

template <class Archive>
void L1TMuonBarrelParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("l1mudttfparams", l1mudttfparams);
    ar & boost::serialization::make_nvp("l1mudttfmasks", l1mudttfmasks);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("fwVersion-", fwVersion_);
    ar & boost::serialization::make_nvp("pnodes-", pnodes_);
    ar & boost::serialization::make_nvp("l1mudttfparams-", l1mudttfparams_);
    ar & boost::serialization::make_nvp("l1mudttfmasks-", l1mudttfmasks_);
    ar & boost::serialization::make_nvp("lutparams-", lutparams_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonBarrelParams);

template <class Archive>
void L1TMuonBarrelParams::LUTParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("pta-lut-", pta_lut_);
    ar & boost::serialization::make_nvp("phi-lut-", phi_lut_);
    ar & boost::serialization::make_nvp("pta-threshold-", pta_threshold_);
    ar & boost::serialization::make_nvp("qp-lut-", qp_lut_);
    ar & boost::serialization::make_nvp("eta-lut-", eta_lut_);
    ar & boost::serialization::make_nvp("ext-lut-", ext_lut_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonBarrelParams::LUTParams);

template <class Archive>
void L1TMuonBarrelParams::LUTParams::extLUT::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("low", low);
    ar & boost::serialization::make_nvp("high", high);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonBarrelParams::LUTParams::extLUT);

template <class Archive>
void L1TMuonBarrelParams::Node::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("LUT-", LUT_);
    ar & boost::serialization::make_nvp("dparams-", dparams_);
    ar & boost::serialization::make_nvp("uparams-", uparams_);
    ar & boost::serialization::make_nvp("iparams-", iparams_);
    ar & boost::serialization::make_nvp("sparams-", sparams_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonBarrelParams::Node);

template <class Archive>
void L1TMuonEndCapForest::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("forest-coll-", forest_coll_);
    ar & boost::serialization::make_nvp("forest-map-", forest_map_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonEndCapForest);

template <class Archive>
void L1TMuonEndCapForest::DTreeNode::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("splitVar", splitVar);
    ar & boost::serialization::make_nvp("splitVal", splitVal);
    ar & boost::serialization::make_nvp("fitVal", fitVal);
    ar & boost::serialization::make_nvp("ileft", ileft);
    ar & boost::serialization::make_nvp("iright", iright);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonEndCapForest::DTreeNode);

template <class Archive>
void L1TMuonEndCapParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("PtAssignVersion-", PtAssignVersion_);
    ar & boost::serialization::make_nvp("firmwareVersion-", firmwareVersion_);
    ar & boost::serialization::make_nvp("PhiMatchWindowSt1-", PhiMatchWindowSt1_);
    ar & boost::serialization::make_nvp("PhiMatchWindowSt2-", PhiMatchWindowSt2_);
    ar & boost::serialization::make_nvp("PhiMatchWindowSt3-", PhiMatchWindowSt3_);
    ar & boost::serialization::make_nvp("PhiMatchWindowSt4-", PhiMatchWindowSt4_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonEndCapParams);

template <class Archive>
void L1TMuonGlobalParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("fwVersion-", fwVersion_);
    ar & boost::serialization::make_nvp("bxMin-", bxMin_);
    ar & boost::serialization::make_nvp("bxMax-", bxMax_);
    ar & boost::serialization::make_nvp("pnodes-", pnodes_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonGlobalParams);

template <class Archive>
void L1TMuonGlobalParams::Node::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("LUT-", LUT_);
    ar & boost::serialization::make_nvp("dparams-", dparams_);
    ar & boost::serialization::make_nvp("uparams-", uparams_);
    ar & boost::serialization::make_nvp("iparams-", iparams_);
    ar & boost::serialization::make_nvp("sparams-", sparams_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonGlobalParams::Node);

template <class Archive>
void L1TMuonOverlapParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("fwVersion-", fwVersion_);
    ar & boost::serialization::make_nvp("pnodes-", pnodes_);
    ar & boost::serialization::make_nvp("layerMap-", layerMap_);
    ar & boost::serialization::make_nvp("refLayerMap-", refLayerMap_);
    ar & boost::serialization::make_nvp("refHitMap-", refHitMap_);
    ar & boost::serialization::make_nvp("globalPhiStart-", globalPhiStart_);
    ar & boost::serialization::make_nvp("layerInputMap-", layerInputMap_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams);

template <class Archive>
void L1TMuonOverlapParams::LayerInputNode::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("iFirstInput", iFirstInput);
    ar & boost::serialization::make_nvp("iLayer", iLayer);
    ar & boost::serialization::make_nvp("nInputs", nInputs);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams::LayerInputNode);

template <class Archive>
void L1TMuonOverlapParams::LayerMapNode::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("hwNumber", hwNumber);
    ar & boost::serialization::make_nvp("logicNumber", logicNumber);
    ar & boost::serialization::make_nvp("bendingLayer", bendingLayer);
    ar & boost::serialization::make_nvp("connectedToLayer", connectedToLayer);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams::LayerMapNode);

template <class Archive>
void L1TMuonOverlapParams::Node::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("LUT-", LUT_);
    ar & boost::serialization::make_nvp("dparams-", dparams_);
    ar & boost::serialization::make_nvp("uparams-", uparams_);
    ar & boost::serialization::make_nvp("iparams-", iparams_);
    ar & boost::serialization::make_nvp("sparams-", sparams_);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams::Node);

template <class Archive>
void L1TMuonOverlapParams::RefHitNode::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("iInput", iInput);
    ar & boost::serialization::make_nvp("iPhiMin", iPhiMin);
    ar & boost::serialization::make_nvp("iPhiMax", iPhiMax);
    ar & boost::serialization::make_nvp("iRefHit", iRefHit);
    ar & boost::serialization::make_nvp("iRefLayer", iRefLayer);
    ar & boost::serialization::make_nvp("iRegion", iRegion);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams::RefHitNode);

template <class Archive>
void L1TMuonOverlapParams::RefLayerMapNode::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("refLayer", refLayer);
    ar & boost::serialization::make_nvp("logicNumber", logicNumber);
}
COND_SERIALIZATION_INSTANTIATE(L1TMuonOverlapParams::RefLayerMapNode);

template <class Archive>
void L1TUtmAlgorithm::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("expression-", expression_);
    ar & boost::serialization::make_nvp("expression-in-condition-", expression_in_condition_);
    ar & boost::serialization::make_nvp("rpn-vector-", rpn_vector_);
    ar & boost::serialization::make_nvp("index-", index_);
    ar & boost::serialization::make_nvp("module-id-", module_id_);
    ar & boost::serialization::make_nvp("module-index-", module_index_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmAlgorithm);

template <class Archive>
void L1TUtmBin::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("hw-index", hw_index);
    ar & boost::serialization::make_nvp("minimum", minimum);
    ar & boost::serialization::make_nvp("maximum", maximum);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmBin);

template <class Archive>
void L1TUtmCondition::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("objects-", objects_);
    ar & boost::serialization::make_nvp("cuts-", cuts_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmCondition);

template <class Archive>
void L1TUtmCut::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("object-type-", object_type_);
    ar & boost::serialization::make_nvp("cut-type-", cut_type_);
    ar & boost::serialization::make_nvp("minimum-", minimum_);
    ar & boost::serialization::make_nvp("maximum-", maximum_);
    ar & boost::serialization::make_nvp("data-", data_);
    ar & boost::serialization::make_nvp("key-", key_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmCut);

template <class Archive>
void L1TUtmCutValue::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("index", index);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmCutValue);

template <class Archive>
void L1TUtmObject::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("comparison-operator-", comparison_operator_);
    ar & boost::serialization::make_nvp("bx-offset-", bx_offset_);
    ar & boost::serialization::make_nvp("threshold-", threshold_);
    ar & boost::serialization::make_nvp("ext-signal-name-", ext_signal_name_);
    ar & boost::serialization::make_nvp("ext-channel-id-", ext_channel_id_);
    ar & boost::serialization::make_nvp("cuts-", cuts_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmObject);

template <class Archive>
void L1TUtmScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("object-", object_);
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("minimum-", minimum_);
    ar & boost::serialization::make_nvp("maximum-", maximum_);
    ar & boost::serialization::make_nvp("step-", step_);
    ar & boost::serialization::make_nvp("n-bits-", n_bits_);
    ar & boost::serialization::make_nvp("bins-", bins_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmScale);

template <class Archive>
void L1TUtmTriggerMenu::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("algorithm-map-", algorithm_map_);
    ar & boost::serialization::make_nvp("condition-map-", condition_map_);
    ar & boost::serialization::make_nvp("scale-map-", scale_map_);
    ar & boost::serialization::make_nvp("external-map-", external_map_);
    ar & boost::serialization::make_nvp("token-to-condition-", token_to_condition_);
    ar & boost::serialization::make_nvp("name-", name_);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("comment-", comment_);
    ar & boost::serialization::make_nvp("datetime-", datetime_);
    ar & boost::serialization::make_nvp("uuid-firmware-", uuid_firmware_);
    ar & boost::serialization::make_nvp("scale-set-name-", scale_set_name_);
    ar & boost::serialization::make_nvp("n-modules-", n_modules_);
    ar & boost::serialization::make_nvp("version", version);
}
COND_SERIALIZATION_INSTANTIATE(L1TUtmTriggerMenu);

template <class Archive>
void L1TriggerKey::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-recordToKey", m_recordToKey);
    ar & boost::serialization::make_nvp("m-tscKey", m_tscKey);
    ar & boost::serialization::make_nvp("m-subsystemKeys", m_subsystemKeys);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKey);

template <class Archive>
void L1TriggerKeyExt::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-recordToKey", m_recordToKey);
    ar & boost::serialization::make_nvp("m-tscKey", m_tscKey);
    ar & boost::serialization::make_nvp("m-subsystemKeys", m_subsystemKeys);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKeyExt);

template <class Archive>
void L1TriggerKeyList::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tscKeyToToken", m_tscKeyToToken);
    ar & boost::serialization::make_nvp("m-recordKeyToken", m_recordKeyToken);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKeyList);

template <class Archive>
void L1TriggerKeyListExt::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tscKeyToToken", m_tscKeyToToken);
    ar & boost::serialization::make_nvp("m-recordKeyToken", m_recordKeyToken);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKeyListExt);

template <class Archive>
void RPCPattern::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-Strips", m_Strips);
    ar & boost::serialization::make_nvp("m-Tower", m_Tower);
    ar & boost::serialization::make_nvp("m-LogSector", m_LogSector);
    ar & boost::serialization::make_nvp("m-LogSegment", m_LogSegment);
    ar & boost::serialization::make_nvp("m-Sign", m_Sign);
    ar & boost::serialization::make_nvp("m-Code", m_Code);
    ar & boost::serialization::make_nvp("m-PatternType", m_PatternType);
    ar & boost::serialization::make_nvp("m-RefGroup", m_RefGroup);
    ar & boost::serialization::make_nvp("m-QualityTabNumber", m_QualityTabNumber);
    ar & boost::serialization::make_nvp("m-Number", m_Number);
}
COND_SERIALIZATION_INSTANTIATE(RPCPattern);

template <class Archive>
void RPCPattern::RPCLogicalStrip::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-StripFrom", m_StripFrom);
    ar & boost::serialization::make_nvp("m-StripTo", m_StripTo);
}
COND_SERIALIZATION_INSTANTIATE(RPCPattern::RPCLogicalStrip);

template <class Archive>
void RPCPattern::TQuality::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-FiredPlanes", m_FiredPlanes);
    ar & boost::serialization::make_nvp("m-QualityTabNumber", m_QualityTabNumber);
    ar & boost::serialization::make_nvp("m-QualityValue", m_QualityValue);
    ar & boost::serialization::make_nvp("m-logsector", m_logsector);
    ar & boost::serialization::make_nvp("m-logsegment", m_logsegment);
    ar & boost::serialization::make_nvp("m-tower", m_tower);
}
COND_SERIALIZATION_INSTANTIATE(RPCPattern::TQuality);

template <class Archive>
void l1t::CaloConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("uconfig-", uconfig_);
    ar & boost::serialization::make_nvp("sconfig-", sconfig_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloConfig);

template <class Archive>
void l1t::CaloParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("pnode-", pnode_);
    ar & boost::serialization::make_nvp("towerp-", towerp_);
    ar & boost::serialization::make_nvp("regionLsb-", regionLsb_);
    ar & boost::serialization::make_nvp("egp-", egp_);
    ar & boost::serialization::make_nvp("taup-", taup_);
    ar & boost::serialization::make_nvp("jetp-", jetp_);
    ar & boost::serialization::make_nvp("etSumLsb-", etSumLsb_);
    ar & boost::serialization::make_nvp("etSumEtaMin-", etSumEtaMin_);
    ar & boost::serialization::make_nvp("etSumEtaMax-", etSumEtaMax_);
    ar & boost::serialization::make_nvp("etSumEtThreshold-", etSumEtThreshold_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams);

template <class Archive>
void l1t::CaloParams::EgParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("lsb-", lsb_);
    ar & boost::serialization::make_nvp("seedThreshold-", seedThreshold_);
    ar & boost::serialization::make_nvp("neighbourThreshold-", neighbourThreshold_);
    ar & boost::serialization::make_nvp("hcalThreshold-", hcalThreshold_);
    ar & boost::serialization::make_nvp("maxHcalEt-", maxHcalEt_);
    ar & boost::serialization::make_nvp("maxPtHOverE-", maxPtHOverE_);
    ar & boost::serialization::make_nvp("minPtJetIsolation-", minPtJetIsolation_);
    ar & boost::serialization::make_nvp("maxPtJetIsolation-", maxPtJetIsolation_);
    ar & boost::serialization::make_nvp("minPtHOverEIsolation-", minPtHOverEIsolation_);
    ar & boost::serialization::make_nvp("maxPtHOverEIsolation-", maxPtHOverEIsolation_);
    ar & boost::serialization::make_nvp("isoAreaNrTowersEta-", isoAreaNrTowersEta_);
    ar & boost::serialization::make_nvp("isoAreaNrTowersPhi-", isoAreaNrTowersPhi_);
    ar & boost::serialization::make_nvp("isoVetoNrTowersPhi-", isoVetoNrTowersPhi_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams::EgParams);

template <class Archive>
void l1t::CaloParams::JetParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("lsb-", lsb_);
    ar & boost::serialization::make_nvp("seedThreshold-", seedThreshold_);
    ar & boost::serialization::make_nvp("neighbourThreshold-", neighbourThreshold_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams::JetParams);

template <class Archive>
void l1t::CaloParams::Node::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("type-", type_);
    ar & boost::serialization::make_nvp("version-", version_);
    ar & boost::serialization::make_nvp("LUT-", LUT_);
    ar & boost::serialization::make_nvp("dparams-", dparams_);
    ar & boost::serialization::make_nvp("uparams-", uparams_);
    ar & boost::serialization::make_nvp("iparams-", iparams_);
    ar & boost::serialization::make_nvp("sparams-", sparams_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams::Node);

template <class Archive>
void l1t::CaloParams::TauParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("lsb-", lsb_);
    ar & boost::serialization::make_nvp("seedThreshold-", seedThreshold_);
    ar & boost::serialization::make_nvp("neighbourThreshold-", neighbourThreshold_);
    ar & boost::serialization::make_nvp("maxPtTauVeto-", maxPtTauVeto_);
    ar & boost::serialization::make_nvp("minPtJetIsolationB-", minPtJetIsolationB_);
    ar & boost::serialization::make_nvp("maxJetIsolationB-", maxJetIsolationB_);
    ar & boost::serialization::make_nvp("maxJetIsolationA-", maxJetIsolationA_);
    ar & boost::serialization::make_nvp("isoEtaMin-", isoEtaMin_);
    ar & boost::serialization::make_nvp("isoEtaMax-", isoEtaMax_);
    ar & boost::serialization::make_nvp("isoAreaNrTowersEta-", isoAreaNrTowersEta_);
    ar & boost::serialization::make_nvp("isoAreaNrTowersPhi-", isoAreaNrTowersPhi_);
    ar & boost::serialization::make_nvp("isoVetoNrTowersPhi-", isoVetoNrTowersPhi_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams::TauParams);

template <class Archive>
void l1t::CaloParams::TowerParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("lsbH-", lsbH_);
    ar & boost::serialization::make_nvp("lsbE-", lsbE_);
    ar & boost::serialization::make_nvp("lsbSum-", lsbSum_);
    ar & boost::serialization::make_nvp("nBitsH-", nBitsH_);
    ar & boost::serialization::make_nvp("nBitsE-", nBitsE_);
    ar & boost::serialization::make_nvp("nBitsSum-", nBitsSum_);
    ar & boost::serialization::make_nvp("nBitsRatio-", nBitsRatio_);
    ar & boost::serialization::make_nvp("maskH-", maskH_);
    ar & boost::serialization::make_nvp("maskE-", maskE_);
    ar & boost::serialization::make_nvp("maskSum-", maskSum_);
    ar & boost::serialization::make_nvp("maskRatio-", maskRatio_);
    ar & boost::serialization::make_nvp("doEncoding-", doEncoding_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::CaloParams::TowerParams);

template <class Archive>
void l1t::LUT::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("nrBitsAddress-", nrBitsAddress_);
    ar & boost::serialization::make_nvp("nrBitsData-", nrBitsData_);
    ar & boost::serialization::make_nvp("addressMask-", addressMask_);
    ar & boost::serialization::make_nvp("dataMask-", dataMask_);
    ar & boost::serialization::make_nvp("data-", data_);
}
COND_SERIALIZATION_INSTANTIATE(l1t::LUT);

