import FWCore.ParameterSet.Config as cms

process = cms.Process("DigFP420Test")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("SimTransport.HectorProducer.HectorTransport_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimRomanPot.SimFP420.FP420Digi_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Cluster_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Track_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Reco_cfi")

process.load("Configuration.EventContent.EventContent_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:Exhume_Hbb.root')
)

process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*', 
        'keep SimTracks_*_*_*', 
        'keep SimVertexs_*_*_*', 
        'keep PSimHits_*_FP420SI_*', 
        'keep DigiCollectionFP420_*_*_*', 
        'keep ClusterCollectionFP420_*_*_*', 
        'keep TrackCollectionFP420_*_*_*', 
        'keep RecoCollectionFP420_*_*_*'),
    fileName = cms.untracked.string('HecExhume_Hbb.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.VtxSmeared*process.LHCTransport*process.g4SimHits*process.mix*process.FP420Digi*process.FP420Cluster*process.FP420Track*process.FP420Reco)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Generator.ApplyPhiCuts = True
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Generator.HepMCProductLabel = 'LHCTransport'


