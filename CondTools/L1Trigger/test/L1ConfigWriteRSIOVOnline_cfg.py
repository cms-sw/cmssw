import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("L1ConfigWriteRSIOVOnline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

options = VarParsing.VarParsing()
options.register('runNumber',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number")
options.register('tagBase',
                 'CRAFT_hlt', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "IOV tags = object_{tagBase}")
options.register('outputDBConnect',
                 'sqlite_file:l1config.db', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Connection string for output DB")
options.register('outputDBAuth',
                 '.', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Authentication path for outputDB")
options.parseArguments()

process.load("CondTools.L1Trigger.L1ConfigRSKeys_cff")

# Get L1TriggerKeyList from DB
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.outputDB = cms.ESSource("PoolDBESSource",
                                process.CondDBCommon,
                                toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1TriggerKeyListRcd'),
    tag = cms.string('L1TriggerKeyList_' + options.tagBase )
    ))
                                )
process.outputDB.connect = options.outputDBConnect
process.outputDB.DBParameters.authenticationPath = options.outputDBAuth

# writer modules
process.load("CondTools.L1Trigger.L1CondDBIOVWriter_cfi")
process.L1CondDBIOVWriter.offlineDB = options.outputDBConnect
process.L1CondDBIOVWriter.offlineAuthentication = options.outputDBAuth
process.L1CondDBIOVWriter.tscKey = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    interval = cms.uint64(1)
)

process.p = cms.Path(process.L1CondDBIOVWriter)

dummyParams = cms.PSet(
    recordInfo = cms.VPSet(cms.PSet(
    record = cms.string('L1JetEtScaleRcd'),
    tag = cms.string('L1JetEtScale_' + options.tagBase),
    type = cms.string('L1CaloEtScale')
    ),
                           cms.PSet(
    record = cms.string('L1EmEtScaleRcd'),
    tag = cms.string('L1EmEtScale_' + options.tagBase),
    type = cms.string('L1CaloEtScale')
    ),
                           cms.PSet(
    record = cms.string('L1MuTriggerScalesRcd'),
    tag = cms.string('L1MuTriggerScales_' + options.tagBase),
    type = cms.string('L1MuTriggerScales')
    ),
                           cms.PSet(
    record = cms.string('L1MuTriggerPtScaleRcd'),
    tag = cms.string('L1MuTriggerPtScale_' + options.tagBase),
    type = cms.string('L1MuTriggerPtScale')
    ),
                           cms.PSet(
    record = cms.string('L1MuGMTScalesRcd'),
    tag = cms.string('L1MuGMTScales_' + options.tagBase),
    type = cms.string('L1MuGMTScales')
    ),
                           cms.PSet(
    record = cms.string('L1MuCSCTFConfigurationRcd'),
    tag = cms.string('L1MuCSCTFConfiguration_' + options.tagBase),
    type = cms.string('L1MuCSCTFConfiguration')
    ),
                           cms.PSet(
    record = cms.string('L1MuCSCTFAlignmentRcd'),
    tag = cms.string('L1MuCSCTFAlignment_' + options.tagBase),
    type = cms.string('L1MuCSCTFAlignment')
    ),
                           cms.PSet(
    record = cms.string('L1MuCSCPtLutRcd'),
    tag = cms.string('L1MuCSCPtLut_' + options.tagBase),
    type = cms.string('L1MuCSCPtLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTEtaPatternLutRcd'),
    tag = cms.string('L1MuDTEtaPatternLut_' + options.tagBase),
    type = cms.string('L1MuDTEtaPatternLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTExtLutRcd'),
    tag = cms.string('L1MuDTExtLut_' + options.tagBase),
    type = cms.string('L1MuDTExtLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTPhiLutRcd'),
    tag = cms.string('L1MuDTPhiLut_' + options.tagBase),
    type = cms.string('L1MuDTPhiLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTPtaLutRcd'),
    tag = cms.string('L1MuDTPtaLut_' + options.tagBase),
    type = cms.string('L1MuDTPtaLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTQualPatternLutRcd'),
    tag = cms.string('L1MuDTQualPatternLut_' + options.tagBase),
    type = cms.string('L1MuDTQualPatternLut')
    ),
                           cms.PSet(
    record = cms.string('L1MuDTTFParametersRcd'),
    tag = cms.string('L1MuDTTFParameters_' + options.tagBase),
    type = cms.string('L1MuDTTFParameters')
    ),
                           cms.PSet(
    record = cms.string('L1RPCConfigRcd'),
    tag = cms.string('L1RPCConfig_' + options.tagBase),
    type = cms.string('L1RPCConfig')
    ),
                           cms.PSet(
    record = cms.string('L1MuGMTParametersRcd'),
    tag = cms.string('L1MuGMTParameters_' + options.tagBase),
    type = cms.string('L1MuGMTParameters')
    ),
                           cms.PSet(
    record = cms.string('L1MuGMTChannelMaskRcd'),
    tag = cms.string('L1MuGMTChannelMask_' + options.tagBase),
    type = cms.string('L1MuGMTChannelMask')
    ),
                           cms.PSet(
    record = cms.string('L1RCTParametersRcd'),
    tag = cms.string('L1RCTParameters_' + options.tagBase),
    type = cms.string('L1RCTParameters')
    ),
                           cms.PSet(
    record = cms.string('L1RCTChannelMaskRcd'),
    tag = cms.string('L1RCTChannelMask_' + options.tagBase),
    type = cms.string('L1RCTChannelMask')
    ),
                           cms.PSet(
    record = cms.string('L1CaloEcalScaleRcd'),
    tag = cms.string('L1CaloEcalScale_' + options.tagBase),
    type = cms.string('L1CaloEcalScale')
    ),
                           cms.PSet(
    record = cms.string('L1CaloHcalScaleRcd'),
    tag = cms.string('L1CaloHcalScale_' + options.tagBase),
    type = cms.string('L1CaloHcalScale')
    ),
                           cms.PSet(
    record = cms.string('L1GctChannelMaskRcd'),
    tag = cms.string('L1GctChannelMask_' + options.tagBase),
    type = cms.string('L1GctChannelMask')
    ),
                           cms.PSet(
    record = cms.string('L1GctHfLutSetupRcd'),
    tag = cms.string('L1GctHfLutSetup_' + options.tagBase),
    type = cms.string('L1GctHfLutSetup')
    ),
                           cms.PSet(
    record = cms.string('L1GctJetFinderParamsRcd'),
    tag = cms.string('L1GctJetFinderParams_' + options.tagBase),
    type = cms.string('L1GctJetFinderParams')
    ),
                           cms.PSet(
    record = cms.string('L1GctJetCalibFunRcd'),
    tag = cms.string('L1GctJetEtCalibrationFunction_' + options.tagBase
                     ),
    type = cms.string('L1GctJetEtCalibrationFunction')
    ),
                           cms.PSet(
    record = cms.string('L1GctJetCounterNegativeEtaRcd'),
    tag = cms.string('L1GctJetCounterNegativeEta_' + options.tagBase),
    type = cms.string('L1GctJetCounterSetup')
    ),
                           cms.PSet(
    record = cms.string('L1GctJetCounterPositiveEtaRcd'),
    tag = cms.string('L1GctJetCounterPositiveEta_' + options.tagBase),
    type = cms.string('L1GctJetCounterSetup')
    ),
                           cms.PSet(
    record = cms.string('L1GtBoardMapsRcd'),
    tag = cms.string('L1GtBoardMaps_' + options.tagBase),
    type = cms.string('L1GtBoardMaps')
    ),
                           cms.PSet(
    record = cms.string('L1GtParametersRcd'),
    tag = cms.string('L1GtParameters_' + options.tagBase),
    type = cms.string('L1GtParameters')
    ),
                           cms.PSet(
    record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
    tag = cms.string('L1GtPrescaleFactorsAlgoTrig_' + options.tagBase),
    type = cms.string('L1GtPrescaleFactors')
    ),
                           cms.PSet(
    record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
    tag = cms.string('L1GtPrescaleFactorsTechTrig_' + options.tagBase),
    type = cms.string('L1GtPrescaleFactors')
    ),
                           cms.PSet(
    record = cms.string('L1GtStableParametersRcd'),
    tag = cms.string('L1GtStableParameters_' + options.tagBase),
    type = cms.string('L1GtStableParameters')
    ),
                           cms.PSet(
    record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
    tag = cms.string('L1GtTriggerMaskAlgoTrig_' + options.tagBase),
    type = cms.string('L1GtTriggerMask')
    ),
                           cms.PSet(
    record = cms.string('L1GtTriggerMaskTechTrigRcd'),
    tag = cms.string('L1GtTriggerMaskTechTrig_' + options.tagBase),
    type = cms.string('L1GtTriggerMask')
    ),
                           cms.PSet(
    record = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
    tag = cms.string('L1GtTriggerMaskVetoAlgoTrig_' + options.tagBase),
    type = cms.string('L1GtTriggerMask')
    ),
                           cms.PSet(
    record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
    tag = cms.string('L1GtTriggerMaskVetoTechTrig_' + options.tagBase),
    type = cms.string('L1GtTriggerMask')
    ),
                           cms.PSet(
    record = cms.string('L1GtTriggerMenuRcd'),
    tag = cms.string('L1GtTriggerMenu_' + options.tagBase),
    type = cms.string('L1GtTriggerMenu')
    ),
                           cms.PSet(
    record = cms.string('L1GtPsbSetupRcd'),
    tag = cms.string('L1GtPsbSetup_' + options.tagBase),
    type = cms.string('L1GtPsbSetup')
    ),
                           cms.PSet(
    record = cms.string('L1CaloGeometryRecord'),
    tag = cms.string('L1CaloGeometry_' + options.tagBase),
    type = cms.string('L1CaloGeometry')
    ))
    )

process.L1CondDBIOVWriter.toPut = dummyParams.recordInfo
