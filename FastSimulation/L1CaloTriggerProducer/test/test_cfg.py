import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FastSimulation.L1CaloTriggerProducer.fastl1calosim_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValZTT-1212543891-STARTUP-2nd-02/0000/40FA3C45-E533-DD11-9B17-000423D98C20.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
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

process.p = cms.Path(process.fastL1CaloSim)
process.e = cms.EndPath(process.Out)
process.Out.fileName = 'test.root'


