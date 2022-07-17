import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


## VarParsing object

options = VarParsing('analysis')

options.register('globaltag', '122X_mcRun3_2021_realistic_v1', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set Global Tag')
options.register('name', 'TEST', VarParsing.multiplicity.singleton, VarParsing.varType.string, 'Set process name')

options.parseArguments()


## Process

process = cms.Process("ExoticaDQM")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag.globaltag = cms.string(options.globaltag)

process.load("DQM.Physics.ExoticaDQM_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.dqmSaver.workflow   = '/Physics/Exotica/WprimeENu'
process.dqmSaver.convention = 'RelVal'


## output
#Needed to the the plots locally

process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('EXOTICA_DQM_' + options.name + '.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_ExoticaDQM'
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
'/store/relval/CMSSW_12_2_0_pre2/RelValZprimeToEE_M-6000_TuneCP5_14TeV-pythia8/GEN-SIM-RECO/122X_mcRun3_2021_realistic_v1_TkmkFitHighStat-v1/2580000/1e0694f9-fbf3-4bd0-8eef-3ec39232e5ad.root'
    )
)

process.load('JetMETCorrections.Configuration.JetCorrectors_cff')


## path definitions
process.p      = cms.Path(
    process.ExoticaDQM

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
                          


