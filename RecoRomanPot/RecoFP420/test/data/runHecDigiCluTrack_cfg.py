import FWCore.ParameterSet.Config as cms

process = cms.Process("DigFP420Test")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

#from SimG4Core.Application.hectorParameter_cfi import *
process.load("SimTransport.HectorProducer.HectorTransport_cfi")
process.transport = cms.Path(process.LHCTransport)

#process.LHCTransport.ZDCTransport = cms.bool(False) ## main flag to set transport for FP420

#process.LHCTransport.Hector.smearEnergy = cms.bool(True)
#process.LHCTransport.Hector.sigmaEnergy    = cms.double(0.001)## GeV

#process.LHCTransport.Hector.smearAng    = cms.bool(True)
#process.LHCTransport.Hector.sigmaSTX    = cms.double(0.01)## urad
#process.LHCTransport.Hector.sigmaSTY    = cms.double(0.01)## urad
## to be changed :
##process.LHCTransport.Hector.Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_7TeV.tfs'),
##process.LHCTransport.Hector.Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_7TeV.tfs'),

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("SimRomanPot.SimFP420.FP420Digi_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Cluster_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Track_cfi")

process.load("RecoRomanPot.RecoFP420.FP420Reco_cfi")
## to be changed :
##process.FP420Reco.Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_7TeV.tfs'),
##process.FP420Reco.Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_7TeV.tfs'),

process.load("Configuration.EventContent.EventContent_cff")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/z/zhokin/fff/FP420development/data/Exhume_Hbb.root')
   fileNames = cms.untracked.vstring('file:ExHuME_CEPHiggsTobb_14TeV_cff_py_GEN_20events.root')
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
    fileName = cms.untracked.string('HecExhume_Hbb_14TeV_20ev.root')
)

process.Timing = cms.Service("Timing")

process.Tracer = cms.Service("Tracer")

process.p1 = cms.Path(process.VtxSmeared*process.LHCTransport*process.g4SimHits*process.mix*process.FP420Digi*process.FP420Cluster*process.FP420Track*process.FP420Reco)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.p1,process.outpath)
process.g4SimHits.Physics.DefaultCutValue =  cms.double(1000.)
process.g4SimHits.UseMagneticField = cms.bool(False)
process.g4SimHits.Generator.ApplyPhiCuts = cms.bool(False)
process.g4SimHits.Generator.ApplyEtaCuts = cms.bool(False)
process.g4SimHits.Generator.HepMCProductLabel = cms.string('LHCTransport')
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)
process.FP420Digi.ApplyTofCut = cms.bool(False)
##   for VtxSmearedGauss:
#process.VtxSmeared.MeanX = 0.0
#process.VtxSmeared.MeanY = 0.0
#process.VtxSmeared.MeanZ = 0.0
##   for VtxSmearedNoSmear:
#process.VtxSmeared.MinX = -0.00000001
#process.VtxSmeared.MaxX = 0.00000001
#process.VtxSmeared.MinY = -0.00000001
#process.VtxSmeared.MaxY = 0.00000001
#process.VtxSmeared.MinZ = -0.00000001
#process.VtxSmeared.MaxZ = 0.00000001


