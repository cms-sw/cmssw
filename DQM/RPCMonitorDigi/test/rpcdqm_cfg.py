import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('DQM',Run2_2017)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet()
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_9_3_0_pre4/RelValSingleMuPt100/GEN-SIM-RECO/93X_mc2017_realistic_v1-v1/00000/2CB3B693-5286-E711-9BD2-003048FFD7AA.root',
        '/store/relval/CMSSW_9_3_0_pre4/RelValSingleMuPt100/GEN-SIM-RECO/93X_mc2017_realistic_v1-v1/00000/38CF6891-5286-E711-A48C-0CC47A78A3B4.root',
    ),
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_9_3_0_pre4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v1-v1/00000/56E52423-4986-E711-A1D3-0025905A6066.root',
        '/store/relval/CMSSW_9_3_0_pre4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v1-v1/00000/7E558B24-4986-E711-A60B-0025905A60CE.root',
        '/store/relval/CMSSW_9_3_0_pre4/RelValSingleMuPt100/GEN-SIM-DIGI-RAW/93X_mc2017_realistic_v1-v1/00000/9A673D21-4986-E711-A989-0025905A6134.root',
    ),
)

process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

process.raw2digi_step = cms.Path(process.RawToDigi)
process.dqmOffline_step = cms.Path(process.DQMOfflineMuon)
#process.dqmHarvest_step = cms.Path(process.DQMHarvestMuon+process.dqmSaver)
process.dqmHarvest_step = cms.Path(process.dqmSaver)

process.schedule = cms.Schedule(
    process.raw2digi_step,
    process.dqmOffline_step,
#    process.dqmHarvesting,
    process.dqmHarvest_step
)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)

from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn
process = setCrossingFrameOn(process)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)

