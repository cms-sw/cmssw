import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkMuDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'FT_53_V21_AN6::All'  #change the latest global tag

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_8_0_14/RelValZMM_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_v15-v1/10000/8461BBB7-9D4E-E611-BA9E-0025905A6088.root'
   )
 )
 
 
runOnData = False
 
 
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
            threshold = cms.untracked.string('DEBUG')
           #threshold = cms.untracked.string('ERROR')
    )
)

process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/Muon')
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

if runOnData:
    process.dqmSaver.saveByRun = cms.untracked.int32(1)
    process.dqmSaver.saveAtJobEnd =cms.untracked.bool(False)
    process.dqmSaver.forceRunNumber = cms.untracked.int32(-1)


process.p = cms.Path(process.ewkMuDQM+process.dqmSaver)
