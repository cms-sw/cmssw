import FWCore.ParameterSet.Config as cms

import os

l1tgmt_basedir = "L1Trigger/L1TMuon/"
lut_dir = os.path.join(l1tgmt_basedir, "data/microgmt_luts/")

microGMTEmulator = cms.EDProducer('l1t::L1TMicroGMTProducer',
    barrelTFInput = cms.InputTag("L1TMicroGMTInputProducer", "BarrelTFMuons"),
    overlapTFInput = cms.InputTag("L1TMicroGMTInputProducer", "OverlapTFMuons"),
    forwardTFInput = cms.InputTag("L1TMicroGMTInputProducer", "ForwardTFMuons"),
    triggerTowerInput = cms.InputTag("L1TMicroGMTInputProducer", "TriggerTowerSums"),

    AbsIsoCheckMemLUTSettings = cms.PSet (
        areaSum_in_width = cms.int32(5),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'AbsIsoCheckMem.lut')),
     ) ,

    IdxSelMemPhiLUTSettings = cms.PSet (
        phi_in_width = cms.int32(10),
        out_width = cms.int32(6),
        filename = cms.string(os.path.join(lut_dir, 'IdxSelMemPhi.lut')),
     ) ,

    FwdPosSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'FwdPosSingleMatchQual.lut')),
     ) ,

    BONegMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'BONegMatchQual.lut')),
     ) ,

    OvlNegSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'OvlNegSingleMatchQual.lut')),
     ) ,

    IdxSelMemEtaLUTSettings = cms.PSet (
        eta_in_width = cms.int32(9),
        out_width = cms.int32(5),
        filename = cms.string(os.path.join(lut_dir, 'IdxSelMemEta.lut')),
     ) ,

    FOPosMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'FOPosMatchQual.lut')),
     ) ,

    FwdNegSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'FwdNegSingleMatchQual.lut')),
     ) ,

    BPhiExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'BPhiExtrapolation.lut')),
     ) ,

    BrlSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'BrlSingleMatchQual.lut')),
     ) ,

    RelIsoCheckMemLUTSettings = cms.PSet (
        areaSum_in_width = cms.int32(5),
        pT_in_width = cms.int32(9),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'RelIsoCheckMem.lut')),
     ) ,

    OPhiExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'OPhiExtrapolation.lut')),
     ) ,

    OvlPosSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'OvlPosSingleMatchQual.lut')),
     ) ,

    FEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'FEtaExtrapolation.lut')),
     ) ,

    BOPosMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'BOPosMatchQual.lut')),
     ) ,

    OEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'OEtaExtrapolation.lut')),
     ) ,

    BEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'BEtaExtrapolation.lut')),
     ) ,

    FPhiExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6),
        pTred_in_width = cms.int32(6),
        out_width = cms.int32(4),
        filename = cms.string(os.path.join(lut_dir, 'FPhiExtrapolation.lut')),
     ) ,

    FONegMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4),
        deltaPhiRed_in_width = cms.int32(3),
        out_width = cms.int32(1),
        filename = cms.string(os.path.join(lut_dir, 'FONegMatchQual.lut')),
     ) ,

    SortRankLUTSettings = cms.PSet (
        pT_in_width = cms.int32(9),
        qual_in_width = cms.int32(4),
        out_width = cms.int32(10),
        filename = cms.string(os.path.join(lut_dir, 'SortRank.lut')),
     )
)

