import FWCore.ParameterSet.Config as cms


# runs particle flow clustering.

process = cms.Process("PFC")

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2022_realistic']


process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring("/store/relval/CMSSW_4_3_0_pre6/RelValTTbar/GEN-SIM-RECO/START43_V3-v1/0085/BC545C44-9F8B-E011-9371-0030486791AA.root") )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")

process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")

# process.particleFlowClusterECAL.verbose = True

process.p = cms.Path(
    process.particleFlowCluster
    )



process.load("Configuration.EventContent.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('clustering.root')
)

# process.reco.outputCommands.append('keep recoPFRecHits_*_*_*')
process.reco.outputCommands.append('keep recoPFClusters_*_*_*')

process.outpath = cms.EndPath( process.reco )

#
