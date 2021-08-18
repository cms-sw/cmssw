import FWCore.ParameterSet.Config as cms

import os

# directory paths to code and external LUTs
l1tgmt_basedir = "L1Trigger/L1TMuon/"
lut_dir = os.path.join(l1tgmt_basedir, "data/microgmt_luts/")

gmtParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonGlobalParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

gmtParams = cms.ESProducer('L1TMuonGlobalParamsESProducer',
    # id for uGMT settings
    uGmtProcessorId = cms.string('ugmt_processor'),
    # paths to external xml online config files
    # for testing purposes or manual generation of conditions from a trigger key
    hwXmlFile = cms.string('L1Trigger/L1TMuon/data/o2o/ugmt/UGMT_HW.xml'),
    topCfgXmlFile = cms.string('L1Trigger/L1TMuon/data/o2o/ugmt/ugmt_top_config_p5.xml'),
    xmlCfgKey = cms.string('TestKey1'),

    # get configuration from OMDS xml files defined above
    # if "True" parameters below this one are ignored
    configFromXml = cms.bool(False),

    # uGMT FW version to be set
    # Format: 0xMMmmvvpp for major.minor.very_minor.patch
    # E.g.: 2.1.0.patch1 translates to 0x02010001
    fwVersion = cms.uint32(0x4010000),

    # uGMT inputs to disable
    # disabled inputs are not used in the algo but are still in the readout
    caloInputsDisable = cms.bool(False), # disables all 28 calo inputs
    bmtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # BMTF 0-11
    omtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # OMTF+0-5, OMTF-0-5
    emtfInputsToDisable = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # EMTF+0-5, EMTF-0-5

    # masked inputs
    # masked inputs are not used in the algo and are not in the readout
    caloInputsMasked = cms.bool(False), # masks all 28 calo inputs
    maskedBmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # BMTF 0-11
    maskedOmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # OMTF+0-5, OMTF-0-5
    maskedEmtfInputs = cms.vuint32(0,0,0,0,0,0,0,0,0,0,0,0), # EMTF+0-5, EMTF-0-5

    # paths to external lookup tables
    # The paths can be empty for MatchQual and SortRAnk LUTs, in which case
    # the LUTs are generated on the fly by the parameters defined below.
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
    BPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BPhiExtrapolation_5eta_7pt_4out_2outshift_20170505.txt')),
    OPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OPhiExtrapolation_5eta_7pt_4out_2outshift_20170505.txt')),
    FPhiExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'EPhiExtrapolation_5eta_7pt_4out_2outshift_20170505.txt')),
    BEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'BEtaExtrapolation_5eta_7pt_4out_0outshift_20170505.txt')),
    OEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'OEtaExtrapolation_5eta_7pt_4out_0outshift_20170505.txt')),
    FEtaExtrapolationLUTPath     = cms.string(os.path.join(lut_dir, 'EEtaExtrapolation_5eta_7pt_4out_0outshift_20170505.txt')),
    SortRankLUTPath              = cms.string(os.path.join(lut_dir, 'SortRank.txt')),

    # parameters for MatchQual LUTs if no external LUT is defined
    # If a path to an external LUT is defined these parameters have no effect
    FwdPosSingleMatchQualLUTMaxDR = cms.double(0.05),
    FwdPosSingleMatchQualLUTfEta  = cms.double(1),
    FwdPosSingleMatchQualLUTfPhi  = cms.double(1),

    FwdNegSingleMatchQualLUTMaxDR = cms.double(0.05),
    FwdNegSingleMatchQualLUTfEta  = cms.double(1),
    FwdNegSingleMatchQualLUTfPhi  = cms.double(1),

    OvlPosSingleMatchQualLUTMaxDR       = cms.double(0.05),
    OvlPosSingleMatchQualLUTfEta        = cms.double(1),
    OvlPosSingleMatchQualLUTfEtaCoarse  = cms.double(1),
    OvlPosSingleMatchQualLUTfPhi        = cms.double(2),

    OvlNegSingleMatchQualLUTMaxDR       = cms.double(0.05),
    OvlNegSingleMatchQualLUTfEta        = cms.double(1),
    OvlNegSingleMatchQualLUTfEtaCoarse  = cms.double(1),
    OvlNegSingleMatchQualLUTfPhi        = cms.double(2),

    BOPosMatchQualLUTMaxDR        = cms.double(0.15),
    BOPosMatchQualLUTfEta         = cms.double(1),
    BOPosMatchQualLUTfEtaCoarse   = cms.double(1),
    BOPosMatchQualLUTfPhi         = cms.double(6),

    BONegMatchQualLUTMaxDR        = cms.double(0.15),
    BONegMatchQualLUTfEta         = cms.double(1),
    BONegMatchQualLUTfEtaCoarse   = cms.double(1),
    BONegMatchQualLUTfPhi         = cms.double(6),

    FOPosMatchQualLUTMaxDR        = cms.double(0.075),
    FOPosMatchQualLUTfEta         = cms.double(1),
    FOPosMatchQualLUTfEtaCoarse   = cms.double(1),
    FOPosMatchQualLUTfPhi         = cms.double(3),

    FONegMatchQualLUTMaxDR        = cms.double(0.075),
    FONegMatchQualLUTfEta         = cms.double(1),
    FONegMatchQualLUTfEtaCoarse   = cms.double(1),
    FONegMatchQualLUTfPhi         = cms.double(3),

    # parameters for SortRank LUT if no path to an external LUT is defined
    # If a path to an external LUT is defined these parameters have no effect
    SortRankLUTPtFactor   = cms.uint32(1), # can be 0 or 1
    SortRankLUTQualFactor = cms.uint32(4), # can be 0 to 34
)

