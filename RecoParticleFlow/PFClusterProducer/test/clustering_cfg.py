import FWCore.ParameterSet.Config as cms

# runs only clustering.
# PFRecHits should be in the input file

process = cms.Process("PFC")

process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring("file:pfrechits.root") )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.load("RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff")

process.particleFlowClusterECAL.verbose = True

process.p1 = cms.Path(
#    process.particleFlowClusterECAL+
#    process.particleFlowClusterHCAL+
    process.particleFlowClusterHFEM
#    process.particleFlowClusterHFHAD
#    process.particleFlowClusterPS 
    )



process.load("Configuration.EventContent.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('clustering_reco.root')
)

process.outpath = cms.EndPath( process.reco )

#
