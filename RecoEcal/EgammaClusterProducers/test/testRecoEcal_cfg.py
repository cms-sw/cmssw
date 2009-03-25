import FWCore.ParameterSet.Config as cms
from RecoEcal.EgammaClusterProducers.geometryForClustering_cff import *

process = cms.Process('testRecoEcal')
process.load('RecoEcal.Configuration.RecoEcal_cff')

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.Geometry_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        '/store/mc/Summer08/Zee_M20/GEN-SIM-RECO/IDEAL_V9_reco-v3/0001/0246B2F4-6FF3-DD11-94D3-0030487F92B5.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_*_*_testRecoEcal'),
    fileName = cms.untracked.string('output_testRecoEcal.root')
)

process.p = cms.Path(process.ecalClusters)
#process.p = cms.Path(hybridClusteringSequence*multi5x5ClusteringSequence)

process.outpath = cms.EndPath(process.out)

