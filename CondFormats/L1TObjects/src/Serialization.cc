
#include "CondFormats/L1TObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void L1CaloEcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_scale);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloEcalScale);

template <class Archive>
void L1CaloEtScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_linScaleMax);
    ar & BOOST_SERIALIZATION_NVP(m_rankScaleMax);
    ar & BOOST_SERIALIZATION_NVP(m_linearLsb);
    ar & BOOST_SERIALIZATION_NVP(m_thresholds);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloEtScale);

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
COND_SERIALIZATION_INSTANTIATE(L1CaloGeometry);

template <class Archive>
void L1CaloHcalScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_scale);
}
COND_SERIALIZATION_INSTANTIATE(L1CaloHcalScale);

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
COND_SERIALIZATION_INSTANTIATE(L1GctChannelMask);

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
COND_SERIALIZATION_INSTANTIATE(L1GctJetFinderParams);

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
COND_SERIALIZATION_INSTANTIATE(L1GtAlgorithm);

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
COND_SERIALIZATION_INSTANTIATE(L1GtBoard);

template <class Archive>
void L1GtBoardMaps::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardMaps);
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
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
    ar & BOOST_SERIALIZATION_NVP(m_correlationParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCaloTemplate);

template <class Archive>
void L1GtCaloTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCaloTemplate::CorrelationParameter);

template <class Archive>
void L1GtCaloTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etThreshold);
    ar & BOOST_SERIALIZATION_NVP(etaRange);
    ar & BOOST_SERIALIZATION_NVP(phiRange);
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
    ar & BOOST_SERIALIZATION_NVP(m_condName);
    ar & BOOST_SERIALIZATION_NVP(m_condCategory);
    ar & BOOST_SERIALIZATION_NVP(m_condType);
    ar & BOOST_SERIALIZATION_NVP(m_objectType);
    ar & BOOST_SERIALIZATION_NVP(m_condGEq);
    ar & BOOST_SERIALIZATION_NVP(m_condChipNr);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCondition);

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
COND_SERIALIZATION_INSTANTIATE(L1GtCorrelationTemplate);

template <class Archive>
void L1GtCorrelationTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtCorrelationTemplate::CorrelationParameter);

template <class Archive>
void L1GtEnergySumTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtEnergySumTemplate);

template <class Archive>
void L1GtEnergySumTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etThreshold);
    ar & BOOST_SERIALIZATION_NVP(energyOverflow);
    ar & BOOST_SERIALIZATION_NVP(phiRange0Word);
    ar & BOOST_SERIALIZATION_NVP(phiRange1Word);
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
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfBitCountsTemplate);

template <class Archive>
void L1GtHfBitCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(countIndex);
    ar & BOOST_SERIALIZATION_NVP(countThreshold);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfBitCountsTemplate::ObjectParameter);

template <class Archive>
void L1GtHfRingEtSumsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfRingEtSumsTemplate);

template <class Archive>
void L1GtHfRingEtSumsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(etSumIndex);
    ar & BOOST_SERIALIZATION_NVP(etSumThreshold);
}
COND_SERIALIZATION_INSTANTIATE(L1GtHfRingEtSumsTemplate::ObjectParameter);

template <class Archive>
void L1GtJetCountsTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtJetCountsTemplate);

template <class Archive>
void L1GtJetCountsTemplate::ObjectParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(countIndex);
    ar & BOOST_SERIALIZATION_NVP(countThreshold);
    ar & BOOST_SERIALIZATION_NVP(countOverflow);
}
COND_SERIALIZATION_INSTANTIATE(L1GtJetCountsTemplate::ObjectParameter);

template <class Archive>
void L1GtMuonTemplate::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("L1GtCondition", boost::serialization::base_object<L1GtCondition>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_objectParameter);
    ar & BOOST_SERIALIZATION_NVP(m_correlationParameter);
}
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate);

template <class Archive>
void L1GtMuonTemplate::CorrelationParameter::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(chargeCorrelation);
    ar & BOOST_SERIALIZATION_NVP(deltaEtaRange);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange0Word);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiRange1Word);
    ar & BOOST_SERIALIZATION_NVP(deltaPhiMaxbits);
}
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate::CorrelationParameter);

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
COND_SERIALIZATION_INSTANTIATE(L1GtMuonTemplate::ObjectParameter);

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
COND_SERIALIZATION_INSTANTIATE(L1GtParameters);

template <class Archive>
void L1GtPrescaleFactors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_prescaleFactors);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPrescaleFactors);

template <class Archive>
void L1GtPsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtBoardSlot);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbCh0SendLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbCh1SendLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbEnableRecLvds);
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbEnableRecSerLink);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPsbConfig);

template <class Archive>
void L1GtPsbSetup::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_gtPsbSetup);
}
COND_SERIALIZATION_INSTANTIATE(L1GtPsbSetup);

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
COND_SERIALIZATION_INSTANTIATE(L1GtStableParameters);

template <class Archive>
void L1GtTriggerMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_triggerMask);
}
COND_SERIALIZATION_INSTANTIATE(L1GtTriggerMask);

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
COND_SERIALIZATION_INSTANTIATE(L1GtTriggerMenu);

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
COND_SERIALIZATION_INSTANTIATE(L1MuBinnedScale);

template <class Archive>
void L1MuCSCPtLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pt_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCPtLut);

template <class Archive>
void L1MuCSCTFAlignment::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(coefficients);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCTFAlignment);

template <class Archive>
void L1MuCSCTFConfiguration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(registers);
}
COND_SERIALIZATION_INSTANTIATE(L1MuCSCTFConfiguration);

template <class Archive>
void L1MuDTEtaPattern::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_id);
    ar & BOOST_SERIALIZATION_NVP(m_wheel);
    ar & BOOST_SERIALIZATION_NVP(m_position);
    ar & BOOST_SERIALIZATION_NVP(m_eta);
    ar & BOOST_SERIALIZATION_NVP(m_qual);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTEtaPattern);

template <class Archive>
void L1MuDTEtaPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTEtaPatternLut);

template <class Archive>
void L1MuDTExtLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ext_lut);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
    ar & BOOST_SERIALIZATION_NVP(nbit_phib);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTExtLut);

template <class Archive>
void L1MuDTExtLut::LUT::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(low);
    ar & BOOST_SERIALIZATION_NVP(high);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTExtLut::LUT);

template <class Archive>
void L1MuDTPhiLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(phi_lut);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
    ar & BOOST_SERIALIZATION_NVP(nbit_phib);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTPhiLut);

template <class Archive>
void L1MuDTPtaLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pta_lut);
    ar & BOOST_SERIALIZATION_NVP(pta_threshold);
    ar & BOOST_SERIALIZATION_NVP(nbit_phi);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTPtaLut);

template <class Archive>
void L1MuDTQualPatternLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lut);
}
COND_SERIALIZATION_INSTANTIATE(L1MuDTQualPatternLut);

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
COND_SERIALIZATION_INSTANTIATE(L1MuDTTFMasks);

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
COND_SERIALIZATION_INSTANTIATE(L1MuDTTFParameters);

template <class Archive>
void L1MuGMTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_SubsystemMask);
}
COND_SERIALIZATION_INSTANTIATE(L1MuGMTChannelMask);

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
COND_SERIALIZATION_INSTANTIATE(L1MuGMTParameters);

template <class Archive>
void L1MuGMTScales::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_ReducedEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_DeltaEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_DeltaPhiScale);
    ar & BOOST_SERIALIZATION_NVP(m_OvlEtaScale);
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
    ar & BOOST_SERIALIZATION_NVP(m_nbits);
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
    ar & BOOST_SERIALIZATION_NVP(m_packing);
    ar & BOOST_SERIALIZATION_NVP(m_NBins);
    ar & BOOST_SERIALIZATION_NVP(m_Scale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuSymmetricBinnedScale);

template <class Archive>
void L1MuTriggerPtScale::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_PtScale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuTriggerPtScale);

template <class Archive>
void L1MuTriggerScales::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_RegionalEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_RegionalEtaScaleCSC);
    ar & BOOST_SERIALIZATION_NVP(m_GMTEtaScale);
    ar & BOOST_SERIALIZATION_NVP(m_PhiScale);
}
COND_SERIALIZATION_INSTANTIATE(L1MuTriggerScales);

template <class Archive>
void L1RCTChannelMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ecalMask);
    ar & BOOST_SERIALIZATION_NVP(hcalMask);
    ar & BOOST_SERIALIZATION_NVP(hfMask);
}
COND_SERIALIZATION_INSTANTIATE(L1RCTChannelMask);

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
COND_SERIALIZATION_INSTANTIATE(L1RCTNoisyChannelMask);

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
COND_SERIALIZATION_INSTANTIATE(L1RCTParameters);

template <class Archive>
void L1RPCBxOrConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstBX);
    ar & BOOST_SERIALIZATION_NVP(m_lastBX);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCBxOrConfig);

template <class Archive>
void L1RPCConeDefinition::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstTower);
    ar & BOOST_SERIALIZATION_NVP(m_lastTower);
    ar & BOOST_SERIALIZATION_NVP(m_LPSizeVec);
    ar & BOOST_SERIALIZATION_NVP(m_ringToTowerVec);
    ar & BOOST_SERIALIZATION_NVP(m_ringToLPVec);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition);

template <class Archive>
void L1RPCConeDefinition::TLPSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_LP);
    ar & BOOST_SERIALIZATION_NVP(m_size);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TLPSize);

template <class Archive>
void L1RPCConeDefinition::TRingToLP::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_etaPart);
    ar & BOOST_SERIALIZATION_NVP(m_hwPlane);
    ar & BOOST_SERIALIZATION_NVP(m_LP);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TRingToLP);

template <class Archive>
void L1RPCConeDefinition::TRingToTower::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_etaPart);
    ar & BOOST_SERIALIZATION_NVP(m_hwPlane);
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeDefinition::TRingToTower);

template <class Archive>
void L1RPCConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_pats);
    ar & BOOST_SERIALIZATION_NVP(m_quals);
    ar & BOOST_SERIALIZATION_NVP(m_ppt);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConfig);

template <class Archive>
void L1RPCHsbConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_hsb0);
    ar & BOOST_SERIALIZATION_NVP(m_hsb1);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCHsbConfig);

template <class Archive>
void L1TriggerKey::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_recordToKey);
    ar & BOOST_SERIALIZATION_NVP(m_tscKey);
    ar & BOOST_SERIALIZATION_NVP(m_subsystemKeys);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKey);

template <class Archive>
void L1TriggerKeyList::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tscKeyToToken);
    ar & BOOST_SERIALIZATION_NVP(m_recordKeyToken);
}
COND_SERIALIZATION_INSTANTIATE(L1TriggerKeyList);

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
COND_SERIALIZATION_INSTANTIATE(RPCPattern);

template <class Archive>
void RPCPattern::RPCLogicalStrip::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_StripFrom);
    ar & BOOST_SERIALIZATION_NVP(m_StripTo);
}
COND_SERIALIZATION_INSTANTIATE(RPCPattern::RPCLogicalStrip);

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
COND_SERIALIZATION_INSTANTIATE(RPCPattern::TQuality);

#include "CondFormats/L1TObjects/src/SerializationManual.h"
