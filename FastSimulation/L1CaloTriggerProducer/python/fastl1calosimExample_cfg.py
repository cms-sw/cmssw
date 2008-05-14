import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi")

process.load("FastSimulation.L1CaloTriggerProducer.fastL1extraParticleMap_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/2006/12/21/mc-onsel-120_Hto2tau_M200/0002/00C73E1E-D3C6-DB11-9043-00E081402F4B.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.Out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep l1extraL1JetParticles_*_*_*', 
        'keep l1extraL1EmParticles_*_*_*', 
        'keep l1extraL1MuonParticles_*_*_*', 
        'keep l1extraL1EtMissParticle_*_*_*', 
        'keep l1extraL1ParticleMaps_*_*_*'),
    fileName = cms.untracked.string('test.root')
)

process.CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

process.p = cms.Path(process.fastL1CaloSim*process.fastL1extraParticleMap)
process.e = cms.EndPath(process.Out)
process.Out.fileName = 'test.root'

