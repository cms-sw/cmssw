import FWCore.ParameterSet.Config as cms

process = cms.Process("testHGCalRecoLocal")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# get timing service up for profiling
#process.TimerService = cms.Service("TimerService")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


# unpacking


# get uncalibrechits with weights method
process.load("RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi")
process.HGCalUncalibRecHit.HGCEEdigiCollection  = 'mix:HGCDigisEE'
process.HGCalUncalibRecHit.HGCHEFdigiCollection = 'mix:HGCDigisHEback'
process.HGCalUncalibRecHit.HGCHEBdigiCollection = 'mix:HGCDigisHEfront'

# get rechits e.g. from the weights
process.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
process.HGCalRecHit.HGCEEuncalibRecHitCollection  = 'HGCalUncalibRecHit:HGCEEUncalibRecHits'
process.HGCalRecHit.HGCHEFuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEFUncalibRecHits'
process.HGCalRecHit.HGCHEBuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEBUncalibRecHits'


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(10) )
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_6_2_0_SLHC26_patch3/RelValSingleGammaPt35Extended/GEN-SIM-DIGI-RAW/PU_PH2_1K_FB_V6_ee18noExt140-v2/00000/EE5E86F1-812C-E511-A91F-00266CFADE34.root'),
    inputCommands = cms.untracked.vstring("keep *_mix_HGC*_*"),
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False)
    )


process.outputmod = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_mix_*_*',
        'keep *_HGCalUncalibRecHit_*_*',
        'keep *_HGCalRecHit_*_*'
        ),
  fileName = cms.untracked.string('testHGCalLocalReco.root')
)

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.HGCalRecoLocal = cms.Sequence(process.HGCalUncalibRecHit +
                                      process.HGCalRecHit
                                      #+process.dumpEv
                                      )

process.p = cms.Path(process.HGCalRecoLocal)

process.ep = cms.EndPath(process.outputmod)
