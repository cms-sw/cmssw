import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )
process.source = cms.Source(
    "PoolSource",
    secondaryFileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_7_1_0_pre3/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/76897917-C0A1-E311-A852-02163E00EA9A.root',
    '/store/relval/CMSSW_7_1_0_pre3/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/7AAC4BC0-C3A1-E311-A8BF-02163E00EAEA.root',
    '/store/relval/CMSSW_7_1_0_pre3/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/9C3661D6-C3A1-E311-BC51-02163E00E997.root',
    '/store/relval/CMSSW_7_1_0_pre3/RelValTTbar_13/GEN-SIM-RECO/POSTLS171_V1-v1/00000/B660879E-B9A1-E311-BFBE-02163E00A0FF.root'
    ),
    fileNames = cms.untracked.vstring(
    '/store/group/phys_egamma/lgray/TTBarPU25ns_reRECO-step4_RECO_EI/step4_RECO_EI-76897917-C0A1-E311-A852-02163E00EA9A.root',
    '/store/group/phys_egamma/lgray/TTBarPU25ns_reRECO-step4_RECO_EI/step4_RECO_EI-7AAC4BC0-C3A1-E311-A8BF-02163E00EAEA.root',
    '/store/group/phys_egamma/lgray/TTBarPU25ns_reRECO-step4_RECO_EI/step4_RECO_EI-9C3661D6-C3A1-E311-BC51-02163E00E997.root',
    '/store/group/phys_egamma/lgray/TTBarPU25ns_reRECO-step4_RECO_EI/step4_RECO_EI-B660879E-B9A1-E311-BFBE-02163E00A0FF.root'
    )
    #dropDescendantsOfDroppedBranches=cms.untracked.bool(False),
    #inputCommands=cms.untracked.vstring(
    #'keep *',
    #'drop recoPFRecHits_*_*_*'
    #)
)

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('clusterValid_ttbar.root')
                                   )

process.preshowerClusterCompare = cms.EDAnalyzer(
    "PFClusterComparator",
    PFClusters = cms.InputTag("particleFlowClusterPS",'','RECO'),
    PFClustersCompare = cms.InputTag("particleFlowClusterPS",'','reRECO'),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(True)
)

process.ecalClusterCompare = cms.EDAnalyzer(
    "PFClusterComparator",
    PFClusters = cms.InputTag("particleFlowClusterECAL",'','RECO'),
    PFClustersCompare = cms.InputTag("particleFlowClusterECAL",'','reRECO'),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(True)
)

process.hcalClusterCompare = cms.EDAnalyzer(
    "PFClusterComparator",
    PFClusters = cms.InputTag("particleFlowClusterHCAL",'','RECO'),
    PFClustersCompare = cms.InputTag("particleFlowClusterHCAL",'','reRECO'),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(True)
)



process.p = cms.Path( process.preshowerClusterCompare +
                      process.ecalClusterCompare      +
                      process.hcalClusterCompare        )


