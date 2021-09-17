import FWCore.ParameterSet.Config as cms

process = cms.Process("L1ConfigValidation")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('prescalesKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Key for GT prescale factors" )
options.register('maskAlgoKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Key for L1GtTriggerMaskAlgoTrigRcd" )
options.register('maskTechKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Key for L1GtTriggerMaskTechTrigRcd" )
options.register('maskVetoAlgoKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Key for L1GtTriggerMaskVetoAlgoTrigRcd" )
options.register('maskVetoTechKey',
                 '', #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Key for L1GtTriggerMaskVetoTechTrigRcd" )
options.register('online',
                 0, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "1 := use online connection string")
options.parseArguments()

# Set up PoolDBOutputService -- needed for l1t::DataWriter
from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum
from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
initL1O2OTags()

if options.online != 1:
    connectStr = 'oracle://cms_orcoff_prod/CMS_COND_31X_L1T'
    authPath = '/afs/cern.ch/cms/DB/conddb'
else:
    connectStr = 'oracle://cms_orcon_prod/CMS_COND_31X_L1T'
    authPath = '/nfshome0/popcondev/conddb_taskWriters/L1T'

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
inputDB = cms.Service("PoolDBOutputService",
                      CondDBSetup,
                      connect = cms.string(connectStr),
                      toPut = cms.VPSet(cms.PSet(
    record = cms.string("L1TriggerKeyRcd"),
    tag = cms.string("L1TriggerKey_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKey ])),
                                        cms.PSet(
    record = cms.string("L1TriggerKeyListRcd"),
    tag = cms.string("L1TriggerKeyList_" + initL1O2OTags.tagBaseVec[ L1CondEnum.L1TriggerKeyList ]))
                                        ))
inputDB.DBParameters.authenticationPath = authPath

from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
initL1Subsystems( tagBaseVec = initL1O2OTags.tagBaseVec )
inputDB.toPut.extend(initL1Subsystems.params.recordInfo)
process.add_(inputDB)
                        
# Source of events, run number is irrelevant
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# Viewer module
process.load('CondTools.L1Trigger.L1GtRunSettingsViewer_cfi')
process.L1GtRunSettingsViewer.prescalesKey = options.prescalesKey
process.L1GtRunSettingsViewer.maskAlgoKey = options.maskAlgoKey
process.L1GtRunSettingsViewer.maskTechKey = options.maskTechKey
process.L1GtRunSettingsViewer.maskVetoAlgoKey = options.maskVetoAlgoKey
process.L1GtRunSettingsViewer.maskVetoTechKey = options.maskVetoTechKey

process.p = cms.Path(process.L1GtRunSettingsViewer)
