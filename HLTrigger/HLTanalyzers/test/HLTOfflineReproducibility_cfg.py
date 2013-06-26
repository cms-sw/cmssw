import FWCore.ParameterSet.Config as cms
#dqm = cms.untracked.bool(True)

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations  = cms.untracked.vstring( 'Info',
                                                                           'Errors',
                                                                           'Warnings',
                                                                           'Debug'),
                                    categories    = cms.untracked.vstring( 'eventNumber',
                                                                           'Root_Information',
                                                                           'EventSetupDependency'
                                                                           ),
                                    Info          = cms.untracked.PSet( threshold = cms.untracked.string('INFO'),
                                                                        Root_Information     = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
                                                                        EventSetupDependency = cms.untracked.PSet( limit = cms.untracked.int32(0) )
                                                                        ),
                                    Errors        = cms.untracked.PSet( threshold = cms.untracked.string('ERROR') ),
                                    Warnings      = cms.untracked.PSet( threshold = cms.untracked.string('WARNING') ),
                                    Debug         = cms.untracked.PSet( threshold =  cms.untracked.string('DEBUG') ),
                                    debugModules  = cms.untracked.vstring('*')
                                    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("HLTrigger.HLTanalyzers.hltOfflineReproducibility_cfi")
from HLTrigger.HLTanalyzers.hltOfflineReproducibility_cfi import *

if DQM:
    process.load("DQMServices.Core.DQMStore_cfg")
    process.load('Configuration.EventContent.EventContent_cff')
    process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['hltonline']
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

#INPUTFILE="rfio:/castor/cern.ch/user/j/jalimena/177139/HT/out_177139_HT_0.root"
INPUTFILE="rfio:/castor/cern.ch/user/j/jalimena/194050/MinimumBias/out_194050_MinimumBias_xxx.root"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(INPUTFILE)
                            )

if DQM:
    process.DQMoutput = cms.OutputModule("PoolOutputModule",
                                         splitLevel = cms.untracked.int32(0),
                                         outputCommands = process.DQMEventContent.outputCommands,
                                         fileName = cms.untracked.string('MyFirstDQMExample.root'),
                                         dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
        )
                                         )

else:
    OUTPUTFILE="../../../OfflineReproducibilityOutput/OfflineReproducibility_194050_MinimumBias.root"
    process.TFileService = cms.Service("TFileService",
                                       fileName = cms.string(OUTPUTFILE)
                                       )

process.p = cms.Path(process.hltOfflineReproducibility)
if DQM:
    process.endjob_step = cms.EndPath(process.endOfProcess)
    process.DQMoutput_step = cms.EndPath(process.DQMoutput)
    process.schedule = cms.Schedule(process.p,process.endjob_step,process.DQMoutput_step)

