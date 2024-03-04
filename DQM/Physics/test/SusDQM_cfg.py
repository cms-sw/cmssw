import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


## VarParsing object

options = VarParsing('analysis')

options.register('globaltag', '140X_dataRun3_HLT_v1', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set Global Tag')
options.register('name', 'TEST', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set process name')

options.parseArguments()


## Process

process = cms.Process("SusDQM")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag.globaltag = cms.string(options.globaltag)

process.load("DQM.Physics.SusDQM_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.dqmSaver.workflow   = '/Physics/Sus/WprimeENu'
process.dqmSaver.convention = 'RelVal'


## output
#Needed to the the plots locally

process.output = cms.OutputModule("PoolOutputModule",
                                  fileName       = cms.untracked.string('SUS_DQM_' + options.name + '.root'),
                                  outputCommands = cms.untracked.vstring(
                                      'drop *_*_*_*',
                                      'keep *_*_*_*SusDQM'
                                  ),
                                  splitLevel     = cms.untracked.int32(0),
                                  dataset = cms.untracked.PSet(
                                      dataTier   = cms.untracked.string(''),
                                      filterName = cms.untracked.string('')
                                  )
                                  )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        '/eos/cms/tier0/store/hidata/HIRun2023A/HIForward18/RAW/v1/000/375/513/00000/f2d28773-c6c0-49d2-ac3f-e3df67c60b92.root'
    )
)

process.load('JetMETCorrections.Configuration.JetCorrectors_cff')


## path definitions
process.p = cms.Path(
    process.susDQM
    
)

process.endjob = cms.Path(
    process.endOfProcess
)

process.fanout = cms.EndPath(
    process.output
)


## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
                          


