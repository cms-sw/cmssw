import FWCore.ParameterSet.Config as cms
process = cms.Process("testRECO")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


# get timing service up for profiling
process.TimerService = cms.Service("TimerService")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# get uncalibrechits with ratio method
from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import ecalGlobalUncalibRecHit

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root')
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root')
)

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer",
    verbose = cms.untracked.bool(True)
)

process.dumpUncalib = cms.EDAnalyzer("EcalUncalibRecHitDump",
    EBUncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EEUncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
)

#process.ecalTestRecoLocal = cms.Sequence(ecalGlobalUncalibRecHit * process.dumpEv)
process.ecalTestRecoLocal = cms.Sequence(ecalGlobalUncalibRecHit)

process.uncalibRecHitOutput = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ecalUncalibHit*_*_*',
        'keep *_ecalRecHit_*_*'
    ),
    fileName = cms.untracked.string('testEcalLocalRecoA.root')
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

process.raw2digi_step = cms.Path(process.RawToDigi)
process.reco_step = cms.Path(process.ecalTestRecoLocal)
process.output_step = cms.EndPath(process.uncalibRecHitOutput)

process.schedule = cms.Schedule(process.raw2digi_step, process.reco_step, process.output_step)
