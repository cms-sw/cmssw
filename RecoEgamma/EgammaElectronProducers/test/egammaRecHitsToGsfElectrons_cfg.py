import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")


process.source = cms.Source("PoolSource",
#    debugVerbosity = cms.untracked.uint32(1),
#    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_4_1_2/RelValZEE/GEN-SIM-RECO/MC_311_V2-v1/0021/74BD0C87-9045-E011-93ED-001A92811724.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string('electrons.root')
)

process.Timing = cms.Service("Timing")

from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff import *
from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *

electronSeeds = cms.Sequence(ecalDrivenElectronSeeds)
electronCkfTrackCandidates.src = cms.InputTag('electronSeeds')

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.iterTracking*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.ecalClusters*process.gsfElectronSequence)

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'MC_311_V2::All'


