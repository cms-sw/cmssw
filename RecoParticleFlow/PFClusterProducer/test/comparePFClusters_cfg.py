import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
    )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:step3_redoClusters.root')
)

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('rePS_histos.root')
                                   )

process.pfClusterComparator = cms.EDAnalyzer("PFClusterComparator",
                                             PFClusters = cms.InputTag("particleFlowClusterPS",'','RECO'),
                                             PFClustersCompare = cms.InputTag("particleFlowClusterPS",'','reCluster'),
                                             verbose = cms.untracked.bool(True),
                                             printBlocks = cms.untracked.bool(True)
                                             )

process.p = cms.Path(process.pfClusterComparator)


