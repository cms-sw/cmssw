process = cms.Process("test")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1000)
)

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("cout")
process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string("DEBUG")    # pring LogDebugs and above
    threshold = cms.untracked.string("INFO")     # print LogInfos and above
#    threshold = cms.untracked.string("WARNING")  # print LogWarnings and above
    )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/102A7D74-300A-DE11-B318-000423D6006E.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/C6345BA8-300A-DE11-A5F2-000423D6CA42.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/E4F6CB66-300A-DE11-BAF4-000423D60FF6.root',
        '/store/relval/CMSSW_3_1_0_pre3/RelValZTT/GEN-SIM-RECO/STARTUP_30X_v1/0001/FE324A2E-800A-DE11-A3A3-000423D99A8E.root'
    )
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CaloEventSetup.CaloTowerConstituents_cfi")

process.load("FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi")

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_*_*_test"),
    fileName = cms.untracked.string('/tmp/chinhan/tree_test.root')
)
process.outpath = cms.EndPath(process.o1)

process.p  = cms.Path(process.fastL1CaloSim)
