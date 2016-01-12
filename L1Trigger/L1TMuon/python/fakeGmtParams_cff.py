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

fakeGmtParams = cms.ESProducer('L1TMuonGlobalParamsESProducer',
    fwVersion = cms.uint32(1),

    AbsIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'AbsIsoCheckMem.txt')),
    RelIsoCheckMemLUTPath        = cms.string(os.path.join(lut_dir, 'RelIsoCheckMem.txt')),
    IdxSelMemPhiLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemPhi.txt')),
    IdxSelMemEtaLUTPath          = cms.string(os.path.join(lut_dir, 'IdxSelMemEta.txt')),
    BrlSingleMatchQualLUTPath    = cms.string(''),
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
    FPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'FPhiExtrapolation.txt')),
    BEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BEtaExtrapolation.txt')),
    OEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OEtaExtrapolation.txt')),
    FEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'FEtaExtrapolation.txt')),
    SortRankLUTPath              = cms.string(os.path.join(lut_dir, 'SortRank.txt')),

    BrlSingleMatchQualLUTMaxDR    = cms.double(0.1),
    FwdPosSingleMatchQualLUTMaxDR = cms.double(0.1),
    FwdNegSingleMatchQualLUTMaxDR = cms.double(0.1),
    OvlPosSingleMatchQualLUTMaxDR = cms.double(0.1),
    OvlNegSingleMatchQualLUTMaxDR = cms.double(0.1),
    BOPosMatchQualLUTMaxDR        = cms.double(0.1),
    BONegMatchQualLUTMaxDR        = cms.double(0.1),
    FOPosMatchQualLUTMaxDR        = cms.double(0.1),
    FONegMatchQualLUTMaxDR        = cms.double(0.1),
)

