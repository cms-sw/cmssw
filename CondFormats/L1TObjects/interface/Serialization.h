#ifndef CondFormats_L1TObjects_Serialization_H
#define CondFormats_L1TObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void L1CaloEcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_scale);
}

template <class Archive>
void L1CaloEtScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_linScaleMax);
    ar & BOOST_SERIALIZATION_NVP(m_rankScaleMax);
    ar & BOOST_SERIALIZATION_NVP(m_linearLsb);
    ar & BOOST_SERIALIZATION_NVP(m_thresholds);
}

template <class Archive>
void L1CaloGeometry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_version);
    ar & BOOST_SERIALIZATION_NVP(m_numberGctEmJetPhiBins);
    ar & BOOST_SERIALIZATION_NVP(m_numberGctEtSumPhiBins);
    ar & BOOST_SERIALIZATION_NVP(m_numberGctHtSumPhiBins);
    ar & BOOST_SERIALIZATION_NVP(m_numberGctCentralEtaBinsPerHalf);
    ar & BOOST_SERIALIZATION_NVP(m_numberGctForwardEtaBinsPerHalf);
    ar & BOOST_SERIALIZATION_NVP(m_etaSignBitOffset);
    ar & BOOST_SERIALIZATION_NVP(m_gctEtaBinBoundaries);
    ar & BOOST_SERIALIZATION_NVP(m_etaBinsPerHalf);
    ar & BOOST_SERIALIZATION_NVP(m_gctEmJetPhiBinWidth);
    ar & BOOST_SERIALIZATION_NVP(m_gctEtSumPhiBinWidth);
    ar & BOOST_SERIALIZATION_NVP(m_gctHtSumPhiBinWidth);
    ar & BOOST_SERIALIZATION_NVP(m_gctEmJetPhiOffset);
    ar & BOOST_SERIALIZATION_NVP(m_gctEtSumPhiOffset);
    ar & BOOST_SERIALIZATION_NVP(m_gctHtSumPhiOffset);
}

template <class Archive>
void L1CaloHcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_scale);
}

template <class Archive>
void L1GctChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(emCrateMask_);
    ar & BOOST_SERIALIZATION_NVP(regionMask_);
    ar & BOOST_SERIALIZATION_NVP(tetMask_);
    ar & BOOST_SERIALIZATION_NVP(metMask_);
    ar & BOOST_SERIALIZATION_NVP(htMask_);
    ar & BOOST_SERIALIZATION_NVP(mhtMask_);
}

template <class Archive>
void L1GctJetFinderParams::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rgnEtLsb_);
    ar & BOOST_SERIALIZATION_NVP(htLsb_);
    ar & BOOST_SERIALIZATION_NVP(cenJetEtSeed_);
    ar & BOOST_SERIALIZATION_NVP(forJetEtSeed_);
    ar & BOOST_SERIALIZATION_NVP(tauJetEtSeed_);
    ar & BOOST_SERIALIZATION_NVP(tauIsoEtThreshold_);
    ar & BOOST_SERIALIZATION_NVP(htJetEtThreshold_);
    ar & BOOST_SERIALIZATION_NVP(mhtJetEtThreshold_);
    ar & BOOST_SERIALIZATION_NVP(cenForJetEtaBoundary_);
    ar & BOOST_SERIALIZATION_NVP(corrType_);
    ar & BOOST_SERIALIZATION_NVP(jetCorrCoeffs_);
    ar & BOOST_SERIALIZATION_NVP(tauCorrCoeffs_);
    ar & BOOST_SERIALIZATION_NVP(convertToEnergy_);
    ar & BOOST_SERIALIZATION_NVP(energyConversionCoeffs_);
}

template <class Archive>
void L1GtAlgorithm::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_algoName);
    ar & BOOST_SERIALIZATION_NVP(m_algoAlias);
    ar & BOOST_SERIALIZATION_NVP(m_algoLogicalExpression);
    ar & BOOST_SERIALIZATION_NVP(m_algoRpnVector);
    ar & BOOST_SERIALIZATION_NVP(m_algoBitNumber);
    ar & BOOST_SERIALIZATION_NVP(m_algoChipNumber);
}

template <class Archive>
void L1GtBoard::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardType);
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardIndex);
    ar & BOOST_SERIALIZATION_NVP(m_gtPositionDaqRecord);
    ar & BOOST_SERIALIZATION_NVP(m_gtPositionEvmRecord);
    ar & BOOST_SERIALIZATION_NVP(m_gtBitDaqActiveBoards);
    ar & BOOST_SERIALIZATION_NVP(m_gtBitEvmActiveBoards);
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardSlot);
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardHexName);
    ar & BOOST_SERIALIZATION_NVP(m_gtQuadInPsb);
    ar & BOOST_SERIALIZATION_NVP(m_gtInputPsbChannels);
}

template <class Archive>
void L1GtBoardMaps::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardMaps);
}

template <class Archive>
void L1GtBptxTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}

template <class Archive>
void L1GtCaloTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
    ar & BOOST_SERIALIZATION_NVP(m_correlationParameter);
}

template <class Archive>
void L1GtCaloTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}

template <class Archive>
void L1GtCaloTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etThreshold);
    ar & BOOST_SERIALIZATION_NVP(etaRange);
    ar & BOOST_SERIALIZATION_NVP(phiRange);
}

template <class Archive>
void L1GtCastorTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}

template <class Archive>
void L1GtCondition::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_condName);
    ar & BOOST_SERIALIZATION_NVP(m_condCategory);
    ar & BOOST_SERIALIZATION_NVP(m_condType);
    ar & BOOST_SERIALIZATION_NVP(m_objectType);
    ar & BOOST_SERIALIZATION_NVP(m_condGEq);
    ar & BOOST_SERIALIZATION_NVP(m_condChipNr);
}

template <class Archive>
void L1GtCorrelationTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_cond0Category);
    ar & BOOST_SERIALIZATION_NVP(m_cond1Category);
    ar & BOOST_SERIALIZATION_NVP(m_cond0Index);
    ar & BOOST_SERIALIZATION_NVP(m_cond1Index);
    ar & BOOST_SERIALIZATION_NVP(m_correlationParameter);
}

template <class Archive>
void L1GtCorrelationTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}

template <class Archive>
void L1GtEnergySumTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}

template <class Archive>
void L1GtEnergySumTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etThreshold);
    ar & BOOST_SERIALIZATION_NVP(energyOverflow);
    ar & BOOST_SERIALIZATION_NVP(phiRange0Word);
    ar & BOOST_SERIALIZATION_NVP(phiRange1Word);
}

template <class Archive>
void L1GtExternalTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
}

template <class Archive>
void L1GtHfBitCountsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}

template <class Archive>
void L1GtHfBitCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(countIndex);
    ar & BOOST_SERIALIZATION_NVP(countThreshold);
}

template <class Archive>
void L1GtHfRingEtSumsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}

template <class Archive>
void L1GtHfRingEtSumsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etSumIndex);
    ar & BOOST_SERIALIZATION_NVP(etSumThreshold);
}

template <class Archive>
void L1GtJetCountsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}

template <class Archive>
void L1GtJetCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(countIndex);
    ar & BOOST_SERIALIZATION_NVP(countThreshold);
    ar & BOOST_SERIALIZATION_NVP(countOverflow);
}

template <class Archive>
void L1GtMuonTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
    ar & BOOST_SERIALIZATION_NVP(m_correlationParameter);
}

template <class Archive>
void L1GtMuonTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chargeCorrelation);
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange0Word);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange1Word);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}

template <class Archive>
void L1GtMuonTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ptHighThreshold);
    ar & BOOST_SERIALIZATION_NVP(ptLowThreshold);
    ar & BOOST_SERIALIZATION_NVP(enableMip);
    ar & BOOST_SERIALIZATION_NVP(enableIso);
    ar & BOOST_SERIALIZATION_NVP(requestIso);
    ar & BOOST_SERIALIZATION_NVP(qualityRange);
    ar & BOOST_SERIALIZATION_NVP(etaRange);
    ar & BOOST_SERIALIZATION_NVP(phiHigh);
    ar & BOOST_SERIALIZATION_NVP(phiLow);
}

template <class Archive>
void L1GtParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_totalBxInEvent);
    ar & BOOST_SERIALIZATION_NVP(m_daqActiveBoards);
    ar & BOOST_SERIALIZATION_NVP(m_evmActiveBoards);
    ar & BOOST_SERIALIZATION_NVP(m_daqNrBxBoard);
    ar & BOOST_SERIALIZATION_NVP(m_evmNrBxBoard);
    ar & BOOST_SERIALIZATION_NVP(m_bstLengthBytes);
}

template <class Archive>
void L1GtPrescaleFactors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_prescaleFactors);
}

template <class Archive>
void L1GtPsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardSlot);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbCh0SendLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbCh1SendLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbEnableRecLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbEnableRecSerLink);
}

template <class Archive>
void L1GtPsbSetup::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbSetup);
}

template <class Archive>
void L1GtStableParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_numberPhysTriggers);
    ar & BOOST_SERIALIZATION_NVP(m_numberPhysTriggersExtended);
    ar & BOOST_SERIALIZATION_NVP(m_numberTechnicalTriggers);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1Mu);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1NoIsoEG);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1IsoEG);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1CenJet);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1ForJet);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1TauJet);
    ar & BOOST_SERIALIZATION_NVP(m_numberL1JetCounts);
    ar & BOOST_SERIALIZATION_NVP(m_numberConditionChips);
    ar & BOOST_SERIALIZATION_NVP(m_pinsOnConditionChip);
    ar & BOOST_SERIALIZATION_NVP(m_orderConditionChip);
    ar & BOOST_SERIALIZATION_NVP(m_numberPsbBoards);
    ar & BOOST_SERIALIZATION_NVP(m_ifCaloEtaNumberBits);
    ar & BOOST_SERIALIZATION_NVP(m_ifMuEtaNumberBits);
    ar & BOOST_SERIALIZATION_NVP(m_wordLength);
    ar & BOOST_SERIALIZATION_NVP(m_unitLength);
}

template <class Archive>
void L1GtTriggerMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_triggerMask);
}

template <class Archive>
void L1GtTriggerMenu::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_triggerMenuInterface);
    ar & BOOST_SERIALIZATION_NVP(m_triggerMenuName);
    ar & BOOST_SERIALIZATION_NVP(m_triggerMenuImplementation);
    ar & BOOST_SERIALIZATION_NVP(m_scaleDbKey);
    ar & BOOST_SERIALIZATION_NVP(m_vecMuonTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecCaloTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecEnergySumTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecJetCountsTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecCastorTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecHfBitCountsTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecHfRingEtSumsTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecBptxTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecExternalTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_vecCorrelationTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_corMuonTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_corCaloTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_corEnergySumTemplate);
    ar & BOOST_SERIALIZATION_NVP(m_algorithmMap);
    ar & BOOST_SERIALIZATION_NVP(m_algorithmAliasMap);
    ar & BOOST_SERIALIZATION_NVP(m_technicalTriggerMap);
}

template <class Archive>
void L1MuBinnedScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuScale", boost::serialization::base_object<L1MuScale>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_nbits);
    ar & BOOST_SERIALIZATION_NVP(m_signedPacking);
    ar & BOOST_SERIALIZATION_NVP(m_NBins);
    ar & BOOST_SERIALIZATION_NVP(m_idxoffset);
    ar & BOOST_SERIALIZATION_NVP(m_Scale);
}

template <class Archive>
void L1MuCSCPtLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pt_lut);
}

template <class Archive>
void L1MuCSCTFAlignment::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(coefficients);
}

template <class Archive>
void L1MuCSCTFConfiguration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(registers);
}

template <class Archive>
void L1MuDTEtaPattern::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_id);
    ar & BOOST_SERIALIZATION_NVP(m_wheel);
    ar & BOOST_SERIALIZATION_NVP(m_position);
    ar & BOOST_SERIALIZATION_NVP(m_eta);
    ar & BOOST_SERIALIZATION_NVP(m_qual);
}

template <class Archive>
void L1MuDTEtaPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lut);
}

template <class Archive>
void L1MuDTExtLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ext_lut);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
    ar & BOOST_SERIALIZATION_NVP(nbit_phib);
}

template <class Archive>
void L1MuDTExtLut::LUT::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}

template <class Archive>
void L1MuDTPhiLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(phi_lut);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
    ar & BOOST_SERIALIZATION_NVP(nbit_phib);
}

template <class Archive>
void L1MuDTPtaLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pta_lut);
    ar & BOOST_SERIALIZATION_NVP(pta_threshold);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
}

template <class Archive>
void L1MuDTQualPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lut);
}

template <class Archive>
void L1MuDTTFMasks::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(inrec_chdis_st1);
    ar & BOOST_SERIALIZATION_NVP(inrec_chdis_st2);
    ar & BOOST_SERIALIZATION_NVP(inrec_chdis_st3);
    ar & BOOST_SERIALIZATION_NVP(inrec_chdis_st4);
    ar & BOOST_SERIALIZATION_NVP(inrec_chdis_csc);
    ar & BOOST_SERIALIZATION_NVP(etsoc_chdis_st1);
    ar & BOOST_SERIALIZATION_NVP(etsoc_chdis_st2);
    ar & BOOST_SERIALIZATION_NVP(etsoc_chdis_st3);
}

template <class Archive>
void L1MuDTTFParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(inrec_qual_st1);
    ar & BOOST_SERIALIZATION_NVP(inrec_qual_st2);
    ar & BOOST_SERIALIZATION_NVP(inrec_qual_st3);
    ar & BOOST_SERIALIZATION_NVP(inrec_qual_st4);
    ar & BOOST_SERIALIZATION_NVP(soc_stdis_n);
    ar & BOOST_SERIALIZATION_NVP(soc_stdis_wl);
    ar & BOOST_SERIALIZATION_NVP(soc_stdis_wr);
    ar & BOOST_SERIALIZATION_NVP(soc_stdis_zl);
    ar & BOOST_SERIALIZATION_NVP(soc_stdis_zr);
    ar & BOOST_SERIALIZATION_NVP(soc_qcut_st1);
    ar & BOOST_SERIALIZATION_NVP(soc_qcut_st2);
    ar & BOOST_SERIALIZATION_NVP(soc_qcut_st4);
    ar & BOOST_SERIALIZATION_NVP(soc_qual_csc);
    ar & BOOST_SERIALIZATION_NVP(soc_run_21);
    ar & BOOST_SERIALIZATION_NVP(soc_nbx_del);
    ar & BOOST_SERIALIZATION_NVP(soc_csc_etacanc);
    ar & BOOST_SERIALIZATION_NVP(soc_openlut_extr);
}

template <class Archive>
void L1MuGMTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_SubsystemMask);
}

template <class Archive>
void L1MuGMTParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_EtaWeight_barrel);
    ar & BOOST_SERIALIZATION_NVP(m_PhiWeight_barrel);
    ar & BOOST_SERIALIZATION_NVP(m_EtaPhiThreshold_barrel);
    ar & BOOST_SERIALIZATION_NVP(m_EtaWeight_endcap);
    ar & BOOST_SERIALIZATION_NVP(m_PhiWeight_endcap);
    ar & BOOST_SERIALIZATION_NVP(m_EtaPhiThreshold_endcap);
    ar & BOOST_SERIALIZATION_NVP(m_EtaWeight_COU);
    ar & BOOST_SERIALIZATION_NVP(m_PhiWeight_COU);
    ar & BOOST_SERIALIZATION_NVP(m_EtaPhiThreshold_COU);
    ar & BOOST_SERIALIZATION_NVP(m_CaloTrigger);
    ar & BOOST_SERIALIZATION_NVP(m_IsolationCellSizeEta);
    ar & BOOST_SERIALIZATION_NVP(m_IsolationCellSizePhi);
    ar & BOOST_SERIALIZATION_NVP(m_DoOvlRpcAnd);
    ar & BOOST_SERIALIZATION_NVP(m_PropagatePhi);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodPhiBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodPhiFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodEtaBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodEtaFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodPtBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodPtFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodChargeBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodChargeFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodMIPBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodMIPFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodMIPSpecialUseANDBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodMIPSpecialUseANDFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodISOBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodISOFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodISOSpecialUseANDBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodISOSpecialUseANDFwd);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodSRKBrl);
    ar & BOOST_SERIALIZATION_NVP(m_MergeMethodSRKFwd);
    ar & BOOST_SERIALIZATION_NVP(m_HaloOverwritesMatchedBrl);
    ar & BOOST_SERIALIZATION_NVP(m_HaloOverwritesMatchedFwd);
    ar & BOOST_SERIALIZATION_NVP(m_SortRankOffsetBrl);
    ar & BOOST_SERIALIZATION_NVP(m_SortRankOffsetFwd);
    ar & BOOST_SERIALIZATION_NVP(m_CDLConfigWordDTCSC);
    ar & BOOST_SERIALIZATION_NVP(m_CDLConfigWordCSCDT);
    ar & BOOST_SERIALIZATION_NVP(m_CDLConfigWordbRPCCSC);
    ar & BOOST_SERIALIZATION_NVP(m_CDLConfigWordfRPCDT);
    ar & BOOST_SERIALIZATION_NVP(m_VersionSortRankEtaQLUT);
    ar & BOOST_SERIALIZATION_NVP(m_VersionLUTs);
}

template <class Archive>
void L1MuGMTScales::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_ReducedEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_DeltaEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_DeltaPhiScale);
    ar & BOOST_SERIALIZATION_NVP(m_OvlEtaScale);
}

template <class Archive>
void L1MuPacking::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void L1MuPseudoSignedPacking::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuPacking", boost::serialization::base_object<L1MuPacking>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_nbits);
}

template <class Archive>
void L1MuScale::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void L1MuSymmetricBinnedScale::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1MuScale", boost::serialization::base_object<L1MuScale>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_packing);
    ar & BOOST_SERIALIZATION_NVP(m_NBins);
    ar & BOOST_SERIALIZATION_NVP(m_Scale);
}

template <class Archive>
void L1MuTriggerPtScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_PtScale);
}

template <class Archive>
void L1MuTriggerScales::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_RegionalEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_RegionalEtaScaleCSC);
    ar & BOOST_SERIALIZATION_NVP(m_GMTEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_PhiScale);
}

template <class Archive>
void L1RCTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ecalMask);
    ar & BOOST_SERIALIZATION_NVP(hcalMask);
    ar & BOOST_SERIALIZATION_NVP(hfMask);
}

template <class Archive>
void L1RCTNoisyChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ecalMask);
    ar & BOOST_SERIALIZATION_NVP(hcalMask);
    ar & BOOST_SERIALIZATION_NVP(hfMask);
    ar & BOOST_SERIALIZATION_NVP(ecalThreshold);
    ar & BOOST_SERIALIZATION_NVP(hcalThreshold);
    ar & BOOST_SERIALIZATION_NVP(hfThreshold);
}

template <class Archive>
void L1RCTParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(eGammaLSB_);
    ar & BOOST_SERIALIZATION_NVP(jetMETLSB_);
    ar & BOOST_SERIALIZATION_NVP(eMinForFGCut_);
    ar & BOOST_SERIALIZATION_NVP(eMaxForFGCut_);
    ar & BOOST_SERIALIZATION_NVP(hOeCut_);
    ar & BOOST_SERIALIZATION_NVP(eMinForHoECut_);
    ar & BOOST_SERIALIZATION_NVP(eMaxForHoECut_);
    ar & BOOST_SERIALIZATION_NVP(hMinForHoECut_);
    ar & BOOST_SERIALIZATION_NVP(eActivityCut_);
    ar & BOOST_SERIALIZATION_NVP(hActivityCut_);
    ar & BOOST_SERIALIZATION_NVP(eicIsolationThreshold_);
    ar & BOOST_SERIALIZATION_NVP(jscQuietThresholdBarrel_);
    ar & BOOST_SERIALIZATION_NVP(jscQuietThresholdEndcap_);
    ar & BOOST_SERIALIZATION_NVP(noiseVetoHB_);
    ar & BOOST_SERIALIZATION_NVP(noiseVetoHEplus_);
    ar & BOOST_SERIALIZATION_NVP(noiseVetoHEminus_);
    ar & BOOST_SERIALIZATION_NVP(useCorrections_);
    ar & BOOST_SERIALIZATION_NVP(eGammaECalScaleFactors_);
    ar & BOOST_SERIALIZATION_NVP(eGammaHCalScaleFactors_);
    ar & BOOST_SERIALIZATION_NVP(jetMETECalScaleFactors_);
    ar & BOOST_SERIALIZATION_NVP(jetMETHCalScaleFactors_);
    ar & BOOST_SERIALIZATION_NVP(ecal_calib_);
    ar & BOOST_SERIALIZATION_NVP(hcal_calib_);
    ar & BOOST_SERIALIZATION_NVP(hcal_high_calib_);
    ar & BOOST_SERIALIZATION_NVP(cross_terms_);
    ar & BOOST_SERIALIZATION_NVP(HoverE_smear_low_);
    ar & BOOST_SERIALIZATION_NVP(HoverE_smear_high_);
}

template <class Archive>
void L1RPCBxOrConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstBX);
    ar & BOOST_SERIALIZATION_NVP(m_lastBX);
}

template <class Archive>
void L1RPCConeDefinition::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstTower);
    ar & BOOST_SERIALIZATION_NVP(m_lastTower);
    ar & BOOST_SERIALIZATION_NVP(m_LPSizeVec);
    ar & BOOST_SERIALIZATION_NVP(m_ringToTowerVec);
    ar & BOOST_SERIALIZATION_NVP(m_ringToLPVec);
}

template <class Archive>
void L1RPCConeDefinition::TLPSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_LP);
    ar & BOOST_SERIALIZATION_NVP(m_size);
}

template <class Archive>
void L1RPCConeDefinition::TRingToLP::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_etaPart);
    ar & BOOST_SERIALIZATION_NVP(m_hwPlane);
    ar & BOOST_SERIALIZATION_NVP(m_LP);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}

template <class Archive>
void L1RPCConeDefinition::TRingToTower::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_etaPart);
    ar & BOOST_SERIALIZATION_NVP(m_hwPlane);
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}

template <class Archive>
void L1RPCConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pats);
    ar & BOOST_SERIALIZATION_NVP(m_quals);
    ar & BOOST_SERIALIZATION_NVP(m_ppt);
}

template <class Archive>
void L1RPCHsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_hsb0);
    ar & BOOST_SERIALIZATION_NVP(m_hsb1);
}

template <class Archive>
void L1TriggerKey::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_recordToKey);
    ar & BOOST_SERIALIZATION_NVP(m_tscKey);
    ar & BOOST_SERIALIZATION_NVP(m_subsystemKeys);
}

template <class Archive>
void L1TriggerKeyList::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tscKeyToToken);
    ar & BOOST_SERIALIZATION_NVP(m_recordKeyToken);
}

template <class Archive>
void RPCPattern::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_Strips);
    ar & BOOST_SERIALIZATION_NVP(m_Tower);
    ar & BOOST_SERIALIZATION_NVP(m_LogSector);
    ar & BOOST_SERIALIZATION_NVP(m_LogSegment);
    ar & BOOST_SERIALIZATION_NVP(m_Sign);
    ar & BOOST_SERIALIZATION_NVP(m_Code);
    ar & BOOST_SERIALIZATION_NVP(m_PatternType);
    ar & BOOST_SERIALIZATION_NVP(m_RefGroup);
    ar & BOOST_SERIALIZATION_NVP(m_QualityTabNumber);
    ar & BOOST_SERIALIZATION_NVP(m_Number);
}

template <class Archive>
void RPCPattern::RPCLogicalStrip::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_StripFrom);
    ar & BOOST_SERIALIZATION_NVP(m_StripTo);
}

template <class Archive>
void RPCPattern::TQuality::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_FiredPlanes);
    ar & BOOST_SERIALIZATION_NVP(m_QualityTabNumber);
    ar & BOOST_SERIALIZATION_NVP(m_QualityValue);
    ar & BOOST_SERIALIZATION_NVP(m_logsector);
    ar & BOOST_SERIALIZATION_NVP(m_logsegment);
    ar & BOOST_SERIALIZATION_NVP(m_tower);
}

namespace cond {
namespace serialization {

template <>
struct access<L1CaloEcalScale>
{
    static bool equal_(const L1CaloEcalScale & first, const L1CaloEcalScale & second)
    {
        return true
            and (equal(first.m_scale, second.m_scale))
        ;
    }
};

template <>
struct access<L1CaloEtScale>
{
    static bool equal_(const L1CaloEtScale & first, const L1CaloEtScale & second)
    {
        return true
            and (equal(first.m_linScaleMax, second.m_linScaleMax))
            and (equal(first.m_rankScaleMax, second.m_rankScaleMax))
            and (equal(first.m_linearLsb, second.m_linearLsb))
            and (equal(first.m_thresholds, second.m_thresholds))
        ;
    }
};

template <>
struct access<L1CaloGeometry>
{
    static bool equal_(const L1CaloGeometry & first, const L1CaloGeometry & second)
    {
        return true
            and (equal(first.m_version, second.m_version))
            and (equal(first.m_numberGctEmJetPhiBins, second.m_numberGctEmJetPhiBins))
            and (equal(first.m_numberGctEtSumPhiBins, second.m_numberGctEtSumPhiBins))
            and (equal(first.m_numberGctHtSumPhiBins, second.m_numberGctHtSumPhiBins))
            and (equal(first.m_numberGctCentralEtaBinsPerHalf, second.m_numberGctCentralEtaBinsPerHalf))
            and (equal(first.m_numberGctForwardEtaBinsPerHalf, second.m_numberGctForwardEtaBinsPerHalf))
            and (equal(first.m_etaSignBitOffset, second.m_etaSignBitOffset))
            and (equal(first.m_gctEtaBinBoundaries, second.m_gctEtaBinBoundaries))
            and (equal(first.m_etaBinsPerHalf, second.m_etaBinsPerHalf))
            and (equal(first.m_gctEmJetPhiBinWidth, second.m_gctEmJetPhiBinWidth))
            and (equal(first.m_gctEtSumPhiBinWidth, second.m_gctEtSumPhiBinWidth))
            and (equal(first.m_gctHtSumPhiBinWidth, second.m_gctHtSumPhiBinWidth))
            and (equal(first.m_gctEmJetPhiOffset, second.m_gctEmJetPhiOffset))
            and (equal(first.m_gctEtSumPhiOffset, second.m_gctEtSumPhiOffset))
            and (equal(first.m_gctHtSumPhiOffset, second.m_gctHtSumPhiOffset))
        ;
    }
};

template <>
struct access<L1CaloHcalScale>
{
    static bool equal_(const L1CaloHcalScale & first, const L1CaloHcalScale & second)
    {
        return true
            and (equal(first.m_scale, second.m_scale))
        ;
    }
};

template <>
struct access<L1GctChannelMask>
{
    static bool equal_(const L1GctChannelMask & first, const L1GctChannelMask & second)
    {
        return true
            and (equal(first.emCrateMask_, second.emCrateMask_))
            and (equal(first.regionMask_, second.regionMask_))
            and (equal(first.tetMask_, second.tetMask_))
            and (equal(first.metMask_, second.metMask_))
            and (equal(first.htMask_, second.htMask_))
            and (equal(first.mhtMask_, second.mhtMask_))
        ;
    }
};

template <>
struct access<L1GctJetFinderParams>
{
    static bool equal_(const L1GctJetFinderParams & first, const L1GctJetFinderParams & second)
    {
        return true
            and (equal(first.rgnEtLsb_, second.rgnEtLsb_))
            and (equal(first.htLsb_, second.htLsb_))
            and (equal(first.cenJetEtSeed_, second.cenJetEtSeed_))
            and (equal(first.forJetEtSeed_, second.forJetEtSeed_))
            and (equal(first.tauJetEtSeed_, second.tauJetEtSeed_))
            and (equal(first.tauIsoEtThreshold_, second.tauIsoEtThreshold_))
            and (equal(first.htJetEtThreshold_, second.htJetEtThreshold_))
            and (equal(first.mhtJetEtThreshold_, second.mhtJetEtThreshold_))
            and (equal(first.cenForJetEtaBoundary_, second.cenForJetEtaBoundary_))
            and (equal(first.corrType_, second.corrType_))
            and (equal(first.jetCorrCoeffs_, second.jetCorrCoeffs_))
            and (equal(first.tauCorrCoeffs_, second.tauCorrCoeffs_))
            and (equal(first.convertToEnergy_, second.convertToEnergy_))
            and (equal(first.energyConversionCoeffs_, second.energyConversionCoeffs_))
        ;
    }
};

template <>
struct access<L1GtAlgorithm>
{
    static bool equal_(const L1GtAlgorithm & first, const L1GtAlgorithm & second)
    {
        return true
            and (equal(first.m_algoName, second.m_algoName))
            and (equal(first.m_algoAlias, second.m_algoAlias))
            and (equal(first.m_algoLogicalExpression, second.m_algoLogicalExpression))
            and (equal(first.m_algoRpnVector, second.m_algoRpnVector))
            and (equal(first.m_algoBitNumber, second.m_algoBitNumber))
            and (equal(first.m_algoChipNumber, second.m_algoChipNumber))
        ;
    }
};

template <>
struct access<L1GtBoard>
{
    static bool equal_(const L1GtBoard & first, const L1GtBoard & second)
    {
        return true
            and (equal(first.m_gtBoardType, second.m_gtBoardType))
            and (equal(first.m_gtBoardIndex, second.m_gtBoardIndex))
            and (equal(first.m_gtPositionDaqRecord, second.m_gtPositionDaqRecord))
            and (equal(first.m_gtPositionEvmRecord, second.m_gtPositionEvmRecord))
            and (equal(first.m_gtBitDaqActiveBoards, second.m_gtBitDaqActiveBoards))
            and (equal(first.m_gtBitEvmActiveBoards, second.m_gtBitEvmActiveBoards))
            and (equal(first.m_gtBoardSlot, second.m_gtBoardSlot))
            and (equal(first.m_gtBoardHexName, second.m_gtBoardHexName))
            and (equal(first.m_gtQuadInPsb, second.m_gtQuadInPsb))
            and (equal(first.m_gtInputPsbChannels, second.m_gtInputPsbChannels))
        ;
    }
};

template <>
struct access<L1GtBoardMaps>
{
    static bool equal_(const L1GtBoardMaps & first, const L1GtBoardMaps & second)
    {
        return true
            and (equal(first.m_gtBoardMaps, second.m_gtBoardMaps))
        ;
    }
};

template <>
struct access<L1GtBptxTemplate>
{
    static bool equal_(const L1GtBptxTemplate & first, const L1GtBptxTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
        ;
    }
};

template <>
struct access<L1GtCaloTemplate>
{
    static bool equal_(const L1GtCaloTemplate & first, const L1GtCaloTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
            and (equal(first.m_correlationParameter, second.m_correlationParameter))
        ;
    }
};

template <>
struct access<L1GtCaloTemplate::CorrelationParameter>
{
    static bool equal_(const L1GtCaloTemplate::CorrelationParameter & first, const L1GtCaloTemplate::CorrelationParameter & second)
    {
        return true
            and (equal(first.deltaEtaRange, second.deltaEtaRange))
            and (equal(first.deltaPhiRange, second.deltaPhiRange))
            and (equal(first.deltaPhiMaxbits, second.deltaPhiMaxbits))
        ;
    }
};

template <>
struct access<L1GtCaloTemplate::ObjectParameter>
{
    static bool equal_(const L1GtCaloTemplate::ObjectParameter & first, const L1GtCaloTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.etThreshold, second.etThreshold))
            and (equal(first.etaRange, second.etaRange))
            and (equal(first.phiRange, second.phiRange))
        ;
    }
};

template <>
struct access<L1GtCastorTemplate>
{
    static bool equal_(const L1GtCastorTemplate & first, const L1GtCastorTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
        ;
    }
};

template <>
struct access<L1GtCondition>
{
    static bool equal_(const L1GtCondition & first, const L1GtCondition & second)
    {
        return true
            and (equal(first.m_condName, second.m_condName))
            and (equal(first.m_condCategory, second.m_condCategory))
            and (equal(first.m_condType, second.m_condType))
            and (equal(first.m_objectType, second.m_objectType))
            and (equal(first.m_condGEq, second.m_condGEq))
            and (equal(first.m_condChipNr, second.m_condChipNr))
        ;
    }
};

template <>
struct access<L1GtCorrelationTemplate>
{
    static bool equal_(const L1GtCorrelationTemplate & first, const L1GtCorrelationTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_cond0Category, second.m_cond0Category))
            and (equal(first.m_cond1Category, second.m_cond1Category))
            and (equal(first.m_cond0Index, second.m_cond0Index))
            and (equal(first.m_cond1Index, second.m_cond1Index))
            and (equal(first.m_correlationParameter, second.m_correlationParameter))
        ;
    }
};

template <>
struct access<L1GtCorrelationTemplate::CorrelationParameter>
{
    static bool equal_(const L1GtCorrelationTemplate::CorrelationParameter & first, const L1GtCorrelationTemplate::CorrelationParameter & second)
    {
        return true
            and (equal(first.deltaEtaRange, second.deltaEtaRange))
            and (equal(first.deltaPhiRange, second.deltaPhiRange))
            and (equal(first.deltaPhiMaxbits, second.deltaPhiMaxbits))
        ;
    }
};

template <>
struct access<L1GtEnergySumTemplate>
{
    static bool equal_(const L1GtEnergySumTemplate & first, const L1GtEnergySumTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
        ;
    }
};

template <>
struct access<L1GtEnergySumTemplate::ObjectParameter>
{
    static bool equal_(const L1GtEnergySumTemplate::ObjectParameter & first, const L1GtEnergySumTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.etThreshold, second.etThreshold))
            and (equal(first.energyOverflow, second.energyOverflow))
            and (equal(first.phiRange0Word, second.phiRange0Word))
            and (equal(first.phiRange1Word, second.phiRange1Word))
        ;
    }
};

template <>
struct access<L1GtExternalTemplate>
{
    static bool equal_(const L1GtExternalTemplate & first, const L1GtExternalTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
        ;
    }
};

template <>
struct access<L1GtHfBitCountsTemplate>
{
    static bool equal_(const L1GtHfBitCountsTemplate & first, const L1GtHfBitCountsTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
        ;
    }
};

template <>
struct access<L1GtHfBitCountsTemplate::ObjectParameter>
{
    static bool equal_(const L1GtHfBitCountsTemplate::ObjectParameter & first, const L1GtHfBitCountsTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.countIndex, second.countIndex))
            and (equal(first.countThreshold, second.countThreshold))
        ;
    }
};

template <>
struct access<L1GtHfRingEtSumsTemplate>
{
    static bool equal_(const L1GtHfRingEtSumsTemplate & first, const L1GtHfRingEtSumsTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
        ;
    }
};

template <>
struct access<L1GtHfRingEtSumsTemplate::ObjectParameter>
{
    static bool equal_(const L1GtHfRingEtSumsTemplate::ObjectParameter & first, const L1GtHfRingEtSumsTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.etSumIndex, second.etSumIndex))
            and (equal(first.etSumThreshold, second.etSumThreshold))
        ;
    }
};

template <>
struct access<L1GtJetCountsTemplate>
{
    static bool equal_(const L1GtJetCountsTemplate & first, const L1GtJetCountsTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
        ;
    }
};

template <>
struct access<L1GtJetCountsTemplate::ObjectParameter>
{
    static bool equal_(const L1GtJetCountsTemplate::ObjectParameter & first, const L1GtJetCountsTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.countIndex, second.countIndex))
            and (equal(first.countThreshold, second.countThreshold))
            and (equal(first.countOverflow, second.countOverflow))
        ;
    }
};

template <>
struct access<L1GtMuonTemplate>
{
    static bool equal_(const L1GtMuonTemplate & first, const L1GtMuonTemplate & second)
    {
        return true
            and (equal(static_cast<const L1GtCondition &>(first), static_cast<const L1GtCondition &>(second)))
            and (equal(first.m_objectParameter, second.m_objectParameter))
            and (equal(first.m_correlationParameter, second.m_correlationParameter))
        ;
    }
};

template <>
struct access<L1GtMuonTemplate::CorrelationParameter>
{
    static bool equal_(const L1GtMuonTemplate::CorrelationParameter & first, const L1GtMuonTemplate::CorrelationParameter & second)
    {
        return true
            and (equal(first.chargeCorrelation, second.chargeCorrelation))
            and (equal(first.deltaEtaRange, second.deltaEtaRange))
            and (equal(first.deltaPhiRange0Word, second.deltaPhiRange0Word))
            and (equal(first.deltaPhiRange1Word, second.deltaPhiRange1Word))
            and (equal(first.deltaPhiMaxbits, second.deltaPhiMaxbits))
        ;
    }
};

template <>
struct access<L1GtMuonTemplate::ObjectParameter>
{
    static bool equal_(const L1GtMuonTemplate::ObjectParameter & first, const L1GtMuonTemplate::ObjectParameter & second)
    {
        return true
            and (equal(first.ptHighThreshold, second.ptHighThreshold))
            and (equal(first.ptLowThreshold, second.ptLowThreshold))
            and (equal(first.enableMip, second.enableMip))
            and (equal(first.enableIso, second.enableIso))
            and (equal(first.requestIso, second.requestIso))
            and (equal(first.qualityRange, second.qualityRange))
            and (equal(first.etaRange, second.etaRange))
            and (equal(first.phiHigh, second.phiHigh))
            and (equal(first.phiLow, second.phiLow))
        ;
    }
};

template <>
struct access<L1GtParameters>
{
    static bool equal_(const L1GtParameters & first, const L1GtParameters & second)
    {
        return true
            and (equal(first.m_totalBxInEvent, second.m_totalBxInEvent))
            and (equal(first.m_daqActiveBoards, second.m_daqActiveBoards))
            and (equal(first.m_evmActiveBoards, second.m_evmActiveBoards))
            and (equal(first.m_daqNrBxBoard, second.m_daqNrBxBoard))
            and (equal(first.m_evmNrBxBoard, second.m_evmNrBxBoard))
            and (equal(first.m_bstLengthBytes, second.m_bstLengthBytes))
        ;
    }
};

template <>
struct access<L1GtPrescaleFactors>
{
    static bool equal_(const L1GtPrescaleFactors & first, const L1GtPrescaleFactors & second)
    {
        return true
            and (equal(first.m_prescaleFactors, second.m_prescaleFactors))
        ;
    }
};

template <>
struct access<L1GtPsbConfig>
{
    static bool equal_(const L1GtPsbConfig & first, const L1GtPsbConfig & second)
    {
        return true
            and (equal(first.m_gtBoardSlot, second.m_gtBoardSlot))
            and (equal(first.m_gtPsbCh0SendLvds, second.m_gtPsbCh0SendLvds))
            and (equal(first.m_gtPsbCh1SendLvds, second.m_gtPsbCh1SendLvds))
            and (equal(first.m_gtPsbEnableRecLvds, second.m_gtPsbEnableRecLvds))
            and (equal(first.m_gtPsbEnableRecSerLink, second.m_gtPsbEnableRecSerLink))
        ;
    }
};

template <>
struct access<L1GtPsbSetup>
{
    static bool equal_(const L1GtPsbSetup & first, const L1GtPsbSetup & second)
    {
        return true
            and (equal(first.m_gtPsbSetup, second.m_gtPsbSetup))
        ;
    }
};

template <>
struct access<L1GtStableParameters>
{
    static bool equal_(const L1GtStableParameters & first, const L1GtStableParameters & second)
    {
        return true
            and (equal(first.m_numberPhysTriggers, second.m_numberPhysTriggers))
            and (equal(first.m_numberPhysTriggersExtended, second.m_numberPhysTriggersExtended))
            and (equal(first.m_numberTechnicalTriggers, second.m_numberTechnicalTriggers))
            and (equal(first.m_numberL1Mu, second.m_numberL1Mu))
            and (equal(first.m_numberL1NoIsoEG, second.m_numberL1NoIsoEG))
            and (equal(first.m_numberL1IsoEG, second.m_numberL1IsoEG))
            and (equal(first.m_numberL1CenJet, second.m_numberL1CenJet))
            and (equal(first.m_numberL1ForJet, second.m_numberL1ForJet))
            and (equal(first.m_numberL1TauJet, second.m_numberL1TauJet))
            and (equal(first.m_numberL1JetCounts, second.m_numberL1JetCounts))
            and (equal(first.m_numberConditionChips, second.m_numberConditionChips))
            and (equal(first.m_pinsOnConditionChip, second.m_pinsOnConditionChip))
            and (equal(first.m_orderConditionChip, second.m_orderConditionChip))
            and (equal(first.m_numberPsbBoards, second.m_numberPsbBoards))
            and (equal(first.m_ifCaloEtaNumberBits, second.m_ifCaloEtaNumberBits))
            and (equal(first.m_ifMuEtaNumberBits, second.m_ifMuEtaNumberBits))
            and (equal(first.m_wordLength, second.m_wordLength))
            and (equal(first.m_unitLength, second.m_unitLength))
        ;
    }
};

template <>
struct access<L1GtTriggerMask>
{
    static bool equal_(const L1GtTriggerMask & first, const L1GtTriggerMask & second)
    {
        return true
            and (equal(first.m_triggerMask, second.m_triggerMask))
        ;
    }
};

template <>
struct access<L1GtTriggerMenu>
{
    static bool equal_(const L1GtTriggerMenu & first, const L1GtTriggerMenu & second)
    {
        return true
            and (equal(first.m_triggerMenuInterface, second.m_triggerMenuInterface))
            and (equal(first.m_triggerMenuName, second.m_triggerMenuName))
            and (equal(first.m_triggerMenuImplementation, second.m_triggerMenuImplementation))
            and (equal(first.m_scaleDbKey, second.m_scaleDbKey))
            and (equal(first.m_vecMuonTemplate, second.m_vecMuonTemplate))
            and (equal(first.m_vecCaloTemplate, second.m_vecCaloTemplate))
            and (equal(first.m_vecEnergySumTemplate, second.m_vecEnergySumTemplate))
            and (equal(first.m_vecJetCountsTemplate, second.m_vecJetCountsTemplate))
            and (equal(first.m_vecCastorTemplate, second.m_vecCastorTemplate))
            and (equal(first.m_vecHfBitCountsTemplate, second.m_vecHfBitCountsTemplate))
            and (equal(first.m_vecHfRingEtSumsTemplate, second.m_vecHfRingEtSumsTemplate))
            and (equal(first.m_vecBptxTemplate, second.m_vecBptxTemplate))
            and (equal(first.m_vecExternalTemplate, second.m_vecExternalTemplate))
            and (equal(first.m_vecCorrelationTemplate, second.m_vecCorrelationTemplate))
            and (equal(first.m_corMuonTemplate, second.m_corMuonTemplate))
            and (equal(first.m_corCaloTemplate, second.m_corCaloTemplate))
            and (equal(first.m_corEnergySumTemplate, second.m_corEnergySumTemplate))
            and (equal(first.m_algorithmMap, second.m_algorithmMap))
            and (equal(first.m_algorithmAliasMap, second.m_algorithmAliasMap))
            and (equal(first.m_technicalTriggerMap, second.m_technicalTriggerMap))
        ;
    }
};

template <>
struct access<L1MuBinnedScale>
{
    static bool equal_(const L1MuBinnedScale & first, const L1MuBinnedScale & second)
    {
        return true
            and (equal(static_cast<const L1MuScale &>(first), static_cast<const L1MuScale &>(second)))
            and (equal(first.m_nbits, second.m_nbits))
            and (equal(first.m_signedPacking, second.m_signedPacking))
            and (equal(first.m_NBins, second.m_NBins))
            and (equal(first.m_idxoffset, second.m_idxoffset))
            and (equal(first.m_Scale, second.m_Scale))
        ;
    }
};

template <>
struct access<L1MuCSCPtLut>
{
    static bool equal_(const L1MuCSCPtLut & first, const L1MuCSCPtLut & second)
    {
        return true
            and (equal(first.pt_lut, second.pt_lut))
        ;
    }
};

template <>
struct access<L1MuCSCTFAlignment>
{
    static bool equal_(const L1MuCSCTFAlignment & first, const L1MuCSCTFAlignment & second)
    {
        return true
            and (equal(first.coefficients, second.coefficients))
        ;
    }
};

template <>
struct access<L1MuCSCTFConfiguration>
{
    static bool equal_(const L1MuCSCTFConfiguration & first, const L1MuCSCTFConfiguration & second)
    {
        return true
            and (equal(first.registers, second.registers))
        ;
    }
};

template <>
struct access<L1MuDTEtaPattern>
{
    static bool equal_(const L1MuDTEtaPattern & first, const L1MuDTEtaPattern & second)
    {
        return true
            and (equal(first.m_id, second.m_id))
            and (equal(first.m_wheel, second.m_wheel))
            and (equal(first.m_position, second.m_position))
            and (equal(first.m_eta, second.m_eta))
            and (equal(first.m_qual, second.m_qual))
        ;
    }
};

template <>
struct access<L1MuDTEtaPatternLut>
{
    static bool equal_(const L1MuDTEtaPatternLut & first, const L1MuDTEtaPatternLut & second)
    {
        return true
            and (equal(first.m_lut, second.m_lut))
        ;
    }
};

template <>
struct access<L1MuDTExtLut>
{
    static bool equal_(const L1MuDTExtLut & first, const L1MuDTExtLut & second)
    {
        return true
            and (equal(first.ext_lut, second.ext_lut))
            and (equal(first.nbit_phi, second.nbit_phi))
            and (equal(first.nbit_phib, second.nbit_phib))
        ;
    }
};

template <>
struct access<L1MuDTExtLut::LUT>
{
    static bool equal_(const L1MuDTExtLut::LUT & first, const L1MuDTExtLut::LUT & second)
    {
        return true
            and (equal(first.low, second.low))
            and (equal(first.high, second.high))
        ;
    }
};

template <>
struct access<L1MuDTPhiLut>
{
    static bool equal_(const L1MuDTPhiLut & first, const L1MuDTPhiLut & second)
    {
        return true
            and (equal(first.phi_lut, second.phi_lut))
            and (equal(first.nbit_phi, second.nbit_phi))
            and (equal(first.nbit_phib, second.nbit_phib))
        ;
    }
};

template <>
struct access<L1MuDTPtaLut>
{
    static bool equal_(const L1MuDTPtaLut & first, const L1MuDTPtaLut & second)
    {
        return true
            and (equal(first.pta_lut, second.pta_lut))
            and (equal(first.pta_threshold, second.pta_threshold))
            and (equal(first.nbit_phi, second.nbit_phi))
        ;
    }
};

template <>
struct access<L1MuDTQualPatternLut>
{
    static bool equal_(const L1MuDTQualPatternLut & first, const L1MuDTQualPatternLut & second)
    {
        return true
            and (equal(first.m_lut, second.m_lut))
        ;
    }
};

template <>
struct access<L1MuDTTFMasks>
{
    static bool equal_(const L1MuDTTFMasks & first, const L1MuDTTFMasks & second)
    {
        return true
            and (equal(first.inrec_chdis_st1, second.inrec_chdis_st1))
            and (equal(first.inrec_chdis_st2, second.inrec_chdis_st2))
            and (equal(first.inrec_chdis_st3, second.inrec_chdis_st3))
            and (equal(first.inrec_chdis_st4, second.inrec_chdis_st4))
            and (equal(first.inrec_chdis_csc, second.inrec_chdis_csc))
            and (equal(first.etsoc_chdis_st1, second.etsoc_chdis_st1))
            and (equal(first.etsoc_chdis_st2, second.etsoc_chdis_st2))
            and (equal(first.etsoc_chdis_st3, second.etsoc_chdis_st3))
        ;
    }
};

template <>
struct access<L1MuDTTFParameters>
{
    static bool equal_(const L1MuDTTFParameters & first, const L1MuDTTFParameters & second)
    {
        return true
            and (equal(first.inrec_qual_st1, second.inrec_qual_st1))
            and (equal(first.inrec_qual_st2, second.inrec_qual_st2))
            and (equal(first.inrec_qual_st3, second.inrec_qual_st3))
            and (equal(first.inrec_qual_st4, second.inrec_qual_st4))
            and (equal(first.soc_stdis_n, second.soc_stdis_n))
            and (equal(first.soc_stdis_wl, second.soc_stdis_wl))
            and (equal(first.soc_stdis_wr, second.soc_stdis_wr))
            and (equal(first.soc_stdis_zl, second.soc_stdis_zl))
            and (equal(first.soc_stdis_zr, second.soc_stdis_zr))
            and (equal(first.soc_qcut_st1, second.soc_qcut_st1))
            and (equal(first.soc_qcut_st2, second.soc_qcut_st2))
            and (equal(first.soc_qcut_st4, second.soc_qcut_st4))
            and (equal(first.soc_qual_csc, second.soc_qual_csc))
            and (equal(first.soc_run_21, second.soc_run_21))
            and (equal(first.soc_nbx_del, second.soc_nbx_del))
            and (equal(first.soc_csc_etacanc, second.soc_csc_etacanc))
            and (equal(first.soc_openlut_extr, second.soc_openlut_extr))
        ;
    }
};

template <>
struct access<L1MuGMTChannelMask>
{
    static bool equal_(const L1MuGMTChannelMask & first, const L1MuGMTChannelMask & second)
    {
        return true
            and (equal(first.m_SubsystemMask, second.m_SubsystemMask))
        ;
    }
};

template <>
struct access<L1MuGMTParameters>
{
    static bool equal_(const L1MuGMTParameters & first, const L1MuGMTParameters & second)
    {
        return true
            and (equal(first.m_EtaWeight_barrel, second.m_EtaWeight_barrel))
            and (equal(first.m_PhiWeight_barrel, second.m_PhiWeight_barrel))
            and (equal(first.m_EtaPhiThreshold_barrel, second.m_EtaPhiThreshold_barrel))
            and (equal(first.m_EtaWeight_endcap, second.m_EtaWeight_endcap))
            and (equal(first.m_PhiWeight_endcap, second.m_PhiWeight_endcap))
            and (equal(first.m_EtaPhiThreshold_endcap, second.m_EtaPhiThreshold_endcap))
            and (equal(first.m_EtaWeight_COU, second.m_EtaWeight_COU))
            and (equal(first.m_PhiWeight_COU, second.m_PhiWeight_COU))
            and (equal(first.m_EtaPhiThreshold_COU, second.m_EtaPhiThreshold_COU))
            and (equal(first.m_CaloTrigger, second.m_CaloTrigger))
            and (equal(first.m_IsolationCellSizeEta, second.m_IsolationCellSizeEta))
            and (equal(first.m_IsolationCellSizePhi, second.m_IsolationCellSizePhi))
            and (equal(first.m_DoOvlRpcAnd, second.m_DoOvlRpcAnd))
            and (equal(first.m_PropagatePhi, second.m_PropagatePhi))
            and (equal(first.m_MergeMethodPhiBrl, second.m_MergeMethodPhiBrl))
            and (equal(first.m_MergeMethodPhiFwd, second.m_MergeMethodPhiFwd))
            and (equal(first.m_MergeMethodEtaBrl, second.m_MergeMethodEtaBrl))
            and (equal(first.m_MergeMethodEtaFwd, second.m_MergeMethodEtaFwd))
            and (equal(first.m_MergeMethodPtBrl, second.m_MergeMethodPtBrl))
            and (equal(first.m_MergeMethodPtFwd, second.m_MergeMethodPtFwd))
            and (equal(first.m_MergeMethodChargeBrl, second.m_MergeMethodChargeBrl))
            and (equal(first.m_MergeMethodChargeFwd, second.m_MergeMethodChargeFwd))
            and (equal(first.m_MergeMethodMIPBrl, second.m_MergeMethodMIPBrl))
            and (equal(first.m_MergeMethodMIPFwd, second.m_MergeMethodMIPFwd))
            and (equal(first.m_MergeMethodMIPSpecialUseANDBrl, second.m_MergeMethodMIPSpecialUseANDBrl))
            and (equal(first.m_MergeMethodMIPSpecialUseANDFwd, second.m_MergeMethodMIPSpecialUseANDFwd))
            and (equal(first.m_MergeMethodISOBrl, second.m_MergeMethodISOBrl))
            and (equal(first.m_MergeMethodISOFwd, second.m_MergeMethodISOFwd))
            and (equal(first.m_MergeMethodISOSpecialUseANDBrl, second.m_MergeMethodISOSpecialUseANDBrl))
            and (equal(first.m_MergeMethodISOSpecialUseANDFwd, second.m_MergeMethodISOSpecialUseANDFwd))
            and (equal(first.m_MergeMethodSRKBrl, second.m_MergeMethodSRKBrl))
            and (equal(first.m_MergeMethodSRKFwd, second.m_MergeMethodSRKFwd))
            and (equal(first.m_HaloOverwritesMatchedBrl, second.m_HaloOverwritesMatchedBrl))
            and (equal(first.m_HaloOverwritesMatchedFwd, second.m_HaloOverwritesMatchedFwd))
            and (equal(first.m_SortRankOffsetBrl, second.m_SortRankOffsetBrl))
            and (equal(first.m_SortRankOffsetFwd, second.m_SortRankOffsetFwd))
            and (equal(first.m_CDLConfigWordDTCSC, second.m_CDLConfigWordDTCSC))
            and (equal(first.m_CDLConfigWordCSCDT, second.m_CDLConfigWordCSCDT))
            and (equal(first.m_CDLConfigWordbRPCCSC, second.m_CDLConfigWordbRPCCSC))
            and (equal(first.m_CDLConfigWordfRPCDT, second.m_CDLConfigWordfRPCDT))
            and (equal(first.m_VersionSortRankEtaQLUT, second.m_VersionSortRankEtaQLUT))
            and (equal(first.m_VersionLUTs, second.m_VersionLUTs))
        ;
    }
};

template <>
struct access<L1MuGMTScales>
{
    static bool equal_(const L1MuGMTScales & first, const L1MuGMTScales & second)
    {
        return true
            and (equal(first.m_ReducedEtaScale, second.m_ReducedEtaScale))
            and (equal(first.m_DeltaEtaScale, second.m_DeltaEtaScale))
            and (equal(first.m_DeltaPhiScale, second.m_DeltaPhiScale))
            and (equal(first.m_OvlEtaScale, second.m_OvlEtaScale))
        ;
    }
};

template <>
struct access<L1MuPacking>
{
    static bool equal_(const L1MuPacking & first, const L1MuPacking & second)
    {
        return true
        ;
    }
};

template <>
struct access<L1MuPseudoSignedPacking>
{
    static bool equal_(const L1MuPseudoSignedPacking & first, const L1MuPseudoSignedPacking & second)
    {
        return true
            and (equal(static_cast<const L1MuPacking &>(first), static_cast<const L1MuPacking &>(second)))
            and (equal(first.m_nbits, second.m_nbits))
        ;
    }
};

template <>
struct access<L1MuScale>
{
    static bool equal_(const L1MuScale & first, const L1MuScale & second)
    {
        return true
        ;
    }
};

template <>
struct access<L1MuSymmetricBinnedScale>
{
    static bool equal_(const L1MuSymmetricBinnedScale & first, const L1MuSymmetricBinnedScale & second)
    {
        return true
            and (equal(static_cast<const L1MuScale &>(first), static_cast<const L1MuScale &>(second)))
            and (equal(first.m_packing, second.m_packing))
            and (equal(first.m_NBins, second.m_NBins))
            and (equal(first.m_Scale, second.m_Scale))
        ;
    }
};

template <>
struct access<L1MuTriggerPtScale>
{
    static bool equal_(const L1MuTriggerPtScale & first, const L1MuTriggerPtScale & second)
    {
        return true
            and (equal(first.m_PtScale, second.m_PtScale))
        ;
    }
};

template <>
struct access<L1MuTriggerScales>
{
    static bool equal_(const L1MuTriggerScales & first, const L1MuTriggerScales & second)
    {
        return true
            and (equal(first.m_RegionalEtaScale, second.m_RegionalEtaScale))
            and (equal(first.m_RegionalEtaScaleCSC, second.m_RegionalEtaScaleCSC))
            and (equal(first.m_GMTEtaScale, second.m_GMTEtaScale))
            and (equal(first.m_PhiScale, second.m_PhiScale))
        ;
    }
};

template <>
struct access<L1RCTChannelMask>
{
    static bool equal_(const L1RCTChannelMask & first, const L1RCTChannelMask & second)
    {
        return true
            and (equal(first.ecalMask, second.ecalMask))
            and (equal(first.hcalMask, second.hcalMask))
            and (equal(first.hfMask, second.hfMask))
        ;
    }
};

template <>
struct access<L1RCTNoisyChannelMask>
{
    static bool equal_(const L1RCTNoisyChannelMask & first, const L1RCTNoisyChannelMask & second)
    {
        return true
            and (equal(first.ecalMask, second.ecalMask))
            and (equal(first.hcalMask, second.hcalMask))
            and (equal(first.hfMask, second.hfMask))
            and (equal(first.ecalThreshold, second.ecalThreshold))
            and (equal(first.hcalThreshold, second.hcalThreshold))
            and (equal(first.hfThreshold, second.hfThreshold))
        ;
    }
};

template <>
struct access<L1RCTParameters>
{
    static bool equal_(const L1RCTParameters & first, const L1RCTParameters & second)
    {
        return true
            and (equal(first.eGammaLSB_, second.eGammaLSB_))
            and (equal(first.jetMETLSB_, second.jetMETLSB_))
            and (equal(first.eMinForFGCut_, second.eMinForFGCut_))
            and (equal(first.eMaxForFGCut_, second.eMaxForFGCut_))
            and (equal(first.hOeCut_, second.hOeCut_))
            and (equal(first.eMinForHoECut_, second.eMinForHoECut_))
            and (equal(first.eMaxForHoECut_, second.eMaxForHoECut_))
            and (equal(first.hMinForHoECut_, second.hMinForHoECut_))
            and (equal(first.eActivityCut_, second.eActivityCut_))
            and (equal(first.hActivityCut_, second.hActivityCut_))
            and (equal(first.eicIsolationThreshold_, second.eicIsolationThreshold_))
            and (equal(first.jscQuietThresholdBarrel_, second.jscQuietThresholdBarrel_))
            and (equal(first.jscQuietThresholdEndcap_, second.jscQuietThresholdEndcap_))
            and (equal(first.noiseVetoHB_, second.noiseVetoHB_))
            and (equal(first.noiseVetoHEplus_, second.noiseVetoHEplus_))
            and (equal(first.noiseVetoHEminus_, second.noiseVetoHEminus_))
            and (equal(first.useCorrections_, second.useCorrections_))
            and (equal(first.eGammaECalScaleFactors_, second.eGammaECalScaleFactors_))
            and (equal(first.eGammaHCalScaleFactors_, second.eGammaHCalScaleFactors_))
            and (equal(first.jetMETECalScaleFactors_, second.jetMETECalScaleFactors_))
            and (equal(first.jetMETHCalScaleFactors_, second.jetMETHCalScaleFactors_))
            and (equal(first.ecal_calib_, second.ecal_calib_))
            and (equal(first.hcal_calib_, second.hcal_calib_))
            and (equal(first.hcal_high_calib_, second.hcal_high_calib_))
            and (equal(first.cross_terms_, second.cross_terms_))
            and (equal(first.HoverE_smear_low_, second.HoverE_smear_low_))
            and (equal(first.HoverE_smear_high_, second.HoverE_smear_high_))
        ;
    }
};

template <>
struct access<L1RPCBxOrConfig>
{
    static bool equal_(const L1RPCBxOrConfig & first, const L1RPCBxOrConfig & second)
    {
        return true
            and (equal(first.m_firstBX, second.m_firstBX))
            and (equal(first.m_lastBX, second.m_lastBX))
        ;
    }
};

template <>
struct access<L1RPCConeDefinition>
{
    static bool equal_(const L1RPCConeDefinition & first, const L1RPCConeDefinition & second)
    {
        return true
            and (equal(first.m_firstTower, second.m_firstTower))
            and (equal(first.m_lastTower, second.m_lastTower))
            and (equal(first.m_LPSizeVec, second.m_LPSizeVec))
            and (equal(first.m_ringToTowerVec, second.m_ringToTowerVec))
            and (equal(first.m_ringToLPVec, second.m_ringToLPVec))
        ;
    }
};

template <>
struct access<L1RPCConeDefinition::TLPSize>
{
    static bool equal_(const L1RPCConeDefinition::TLPSize & first, const L1RPCConeDefinition::TLPSize & second)
    {
        return true
            and (equal(first.m_tower, second.m_tower))
            and (equal(first.m_LP, second.m_LP))
            and (equal(first.m_size, second.m_size))
        ;
    }
};

template <>
struct access<L1RPCConeDefinition::TRingToLP>
{
    static bool equal_(const L1RPCConeDefinition::TRingToLP & first, const L1RPCConeDefinition::TRingToLP & second)
    {
        return true
            and (equal(first.m_etaPart, second.m_etaPart))
            and (equal(first.m_hwPlane, second.m_hwPlane))
            and (equal(first.m_LP, second.m_LP))
            and (equal(first.m_index, second.m_index))
        ;
    }
};

template <>
struct access<L1RPCConeDefinition::TRingToTower>
{
    static bool equal_(const L1RPCConeDefinition::TRingToTower & first, const L1RPCConeDefinition::TRingToTower & second)
    {
        return true
            and (equal(first.m_etaPart, second.m_etaPart))
            and (equal(first.m_hwPlane, second.m_hwPlane))
            and (equal(first.m_tower, second.m_tower))
            and (equal(first.m_index, second.m_index))
        ;
    }
};

template <>
struct access<L1RPCConfig>
{
    static bool equal_(const L1RPCConfig & first, const L1RPCConfig & second)
    {
        return true
            and (equal(first.m_pats, second.m_pats))
            and (equal(first.m_quals, second.m_quals))
            and (equal(first.m_ppt, second.m_ppt))
        ;
    }
};

template <>
struct access<L1RPCHsbConfig>
{
    static bool equal_(const L1RPCHsbConfig & first, const L1RPCHsbConfig & second)
    {
        return true
            and (equal(first.m_hsb0, second.m_hsb0))
            and (equal(first.m_hsb1, second.m_hsb1))
        ;
    }
};

template <>
struct access<L1TriggerKey>
{
    static bool equal_(const L1TriggerKey & first, const L1TriggerKey & second)
    {
        return true
            and (equal(first.m_recordToKey, second.m_recordToKey))
            and (equal(first.m_tscKey, second.m_tscKey))
            and (equal(first.m_subsystemKeys, second.m_subsystemKeys))
        ;
    }
};

template <>
struct access<L1TriggerKeyList>
{
    static bool equal_(const L1TriggerKeyList & first, const L1TriggerKeyList & second)
    {
        return true
            and (equal(first.m_tscKeyToToken, second.m_tscKeyToToken))
            and (equal(first.m_recordKeyToken, second.m_recordKeyToken))
        ;
    }
};

template <>
struct access<RPCPattern>
{
    static bool equal_(const RPCPattern & first, const RPCPattern & second)
    {
        return true
            and (equal(first.m_Strips, second.m_Strips))
            and (equal(first.m_Tower, second.m_Tower))
            and (equal(first.m_LogSector, second.m_LogSector))
            and (equal(first.m_LogSegment, second.m_LogSegment))
            and (equal(first.m_Sign, second.m_Sign))
            and (equal(first.m_Code, second.m_Code))
            and (equal(first.m_PatternType, second.m_PatternType))
            and (equal(first.m_RefGroup, second.m_RefGroup))
            and (equal(first.m_QualityTabNumber, second.m_QualityTabNumber))
            and (equal(first.m_Number, second.m_Number))
        ;
    }
};

template <>
struct access<RPCPattern::RPCLogicalStrip>
{
    static bool equal_(const RPCPattern::RPCLogicalStrip & first, const RPCPattern::RPCLogicalStrip & second)
    {
        return true
            and (equal(first.m_StripFrom, second.m_StripFrom))
            and (equal(first.m_StripTo, second.m_StripTo))
        ;
    }
};

template <>
struct access<RPCPattern::TQuality>
{
    static bool equal_(const RPCPattern::TQuality & first, const RPCPattern::TQuality & second)
    {
        return true
            and (equal(first.m_FiredPlanes, second.m_FiredPlanes))
            and (equal(first.m_QualityTabNumber, second.m_QualityTabNumber))
            and (equal(first.m_QualityValue, second.m_QualityValue))
            and (equal(first.m_logsector, second.m_logsector))
            and (equal(first.m_logsegment, second.m_logsegment))
            and (equal(first.m_tower, second.m_tower))
        ;
    }
};

}
}

#endif
