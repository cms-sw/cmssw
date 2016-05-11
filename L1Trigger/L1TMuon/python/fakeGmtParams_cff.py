import FWCore.ParameterSet.Config as cms

import os

l1tgmt_basedir = "L1Trigger/L1TMuon/"
lut_dir = os.path.join(l1tgmt_basedir, "data/microgmt_luts/")

gmtParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonGlobalParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

gmtParams = cms.ESProducer('L1TMuonGlobalParamsESProducer',
    #fwVersion = cms.uint32(1),
    fwVersion = cms.uint32(0x2020000),

    # uGMT inputs to disable
    # disabled inputs are not used in the algo but are still in the readout
    caloInputsDisable = cms.bool(False),
    bmtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # BMTF 0-11
    omtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # OMTF+0-5, OMTF-0-5
    emtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # EMTF+0-5, EMTF-0-5

    # masked inputs
    # masked inputs are not used in the algo and are not in the readout
    caloInputsMasked = cms.bool(False),
    maskedBmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # BMTF 0-11
    maskedOmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # OMTF+0-5, OMTF-0-5
    maskedEmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # EMTF+0-5, EMTF-0-5

    AbsIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'AbsIsoCheckMem.txt')),
    RelIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'RelIsoCheckMem.txt')),
    IdxSelMemPhiLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemPhi.txt')),
    IdxSelMemEtaLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemEta.txt')),
    FwdPosSingleMatchQualLUTPath = cms.string(''),
    FwdNegSingleMatchQualLUTPath = cms.string(''),
    OvlPosSingleMatchQualLUTPath = cms.string(''),
    OvlNegSingleMatchQualLUTPath = cms.string(''),
    BOPosMatchQualLUTPath        = cms.string(''),
    BONegMatchQualLUTPath        = cms.string(''),
    FOPosMatchQualLUTPath        = cms.string(''),
    FONegMatchQualLUTPath        = cms.string(''),
    BPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BPhiExtrapolation.txt')),
    OPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OPhiExtrapolation.txt')),
    FPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'EPhiExtrapolation.txt')),
    BEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BEtaExtrapolation.txt')),
    OEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OEtaExtrapolation.txt')),
    FEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'EEtaExtrapolation.txt')),
    SortRankLUTPath              = cms.string(os.path.join(lut_dir, 'SortRank.txt')),

    FwdPosSingleMatchQualLUTMaxDR = cms.double(0.01),
    FwdPosSingleMatchQualLUTfEta  = cms.double(1),
    FwdPosSingleMatchQualLUTfPhi  = cms.double(1),

    FwdNegSingleMatchQualLUTMaxDR = cms.double(0.01),
    FwdNegSingleMatchQualLUTfEta  = cms.double(1),
    FwdNegSingleMatchQualLUTfPhi  = cms.double(1),

    OvlPosSingleMatchQualLUTMaxDR       = cms.double(0.15),
    OvlPosSingleMatchQualLUTfEta        = cms.double(1),
    OvlPosSingleMatchQualLUTfEtaCoarse  = cms.double(1),
    OvlPosSingleMatchQualLUTfPhi        = cms.double(3),

    OvlNegSingleMatchQualLUTMaxDR       = cms.double(0.15),
    OvlNegSingleMatchQualLUTfEta        = cms.double(1),
    OvlNegSingleMatchQualLUTfEtaCoarse  = cms.double(1),
    OvlNegSingleMatchQualLUTfPhi        = cms.double(3),

    BOPosMatchQualLUTMaxDR        = cms.double(0.1),
    BOPosMatchQualLUTfEta         = cms.double(1./3),
    BOPosMatchQualLUTfEtaCoarse   = cms.double(1./3),
    BOPosMatchQualLUTfPhi         = cms.double(2),

    BONegMatchQualLUTMaxDR        = cms.double(0.1),
    BONegMatchQualLUTfEta         = cms.double(1./3),
    BONegMatchQualLUTfEtaCoarse   = cms.double(1./3),
    BONegMatchQualLUTfPhi         = cms.double(2),

    FOPosMatchQualLUTMaxDR        = cms.double(0.2),
    FOPosMatchQualLUTfEta         = cms.double(1),
    FOPosMatchQualLUTfEtaCoarse   = cms.double(1),
    FOPosMatchQualLUTfPhi         = cms.double(4),

    FONegMatchQualLUTMaxDR        = cms.double(0.2),
    FONegMatchQualLUTfEta         = cms.double(1),
    FONegMatchQualLUTfEtaCoarse   = cms.double(1),
    FONegMatchQualLUTfPhi         = cms.double(4),

    SortRankLUTPtFactor   = cms.uint32(1), # can be 0 or 1
    SortRankLUTQualFactor = cms.uint32(4), # can be 0 to 34
)

