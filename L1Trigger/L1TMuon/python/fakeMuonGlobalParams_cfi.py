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
    fwVersion = cms.uint32(1),

    AbsIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'AbsIsoCheckMem.lut')),
    RelIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'RelIsoCheckMem.lut')),
    IdxSelMemPhiLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemPhi.lut')),
    IdxSelMemEtaLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemEta.lut')),
    BrlSingleMatchQualLUTPath    = cms.string(os.path.join(lut_dir, 'BrlSingleMatchQual.lut')),
    FwdPosSingleMatchQualLUTPath = cms.string(os.path.join(lut_dir, 'FwdPosSingleMatchQual.lut')),
    FwdNegSingleMatchQualLUTPath = cms.string(os.path.join(lut_dir, 'FwdNegSingleMatchQual.lut')),
    OvlPosSingleMatchQualLUTPath = cms.string(os.path.join(lut_dir, 'OvlPosSingleMatchQual.lut')),
    OvlNegSingleMatchQualLUTPath = cms.string(os.path.join(lut_dir, 'OvlNegSingleMatchQual.lut')),
    BOPosMatchQualLUTPath        = cms.string(os.path.join(lut_dir, 'BOPosMatchQual.lut')),
    BONegMatchQualLUTPath        = cms.string(os.path.join(lut_dir, 'BONegMatchQual.lut')),
    FOPosMatchQualLUTPath        = cms.string(os.path.join(lut_dir, 'FOPosMatchQual.lut')),
    FONegMatchQualLUTPath        = cms.string(os.path.join(lut_dir, 'FONegMatchQual.lut')),
    BPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BPhiExtrapolation.lut')),
    OPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OPhiExtrapolation.lut')),
    FPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'FPhiExtrapolation.lut')),
    BEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BEtaExtrapolation.lut')),
    OEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OEtaExtrapolation.lut')),
    FEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'FEtaExtrapolation.lut')),
    SortRankLUTPath              = cms.string(os.path.join(lut_dir, 'SortRank.lut')),
)

