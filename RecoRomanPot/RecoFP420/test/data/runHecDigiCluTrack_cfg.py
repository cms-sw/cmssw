import FWCore.ParameterSet.Config as cms

process = cms.Process("DigFP420Test")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

#process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
# event vertex smearing - applies only once (internal check)
# Note : all internal generators will always do (0,0,0) vertex
#
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")


process.load("Geometry.FP420CommonData.FP420GeometryXML_cfi")

process.load("SimG4Core.Application.g4SimHits_cfi")

#from SimG4Core.Application.hectorParameter_cfi import *
process.load("SimTransport.HectorProducer.HectorTransport_cfi")
process.transport = cms.Path(process.LHCTransport)

process.LHCTransport.ZDCTransport = cms.bool(False) ## main flag to set transport for FP420

process.LHCTransport.Hector.smearEnergy = cms.bool(True)
#process.LHCTransport.Hector.sigmaEnergy    = cms.double(0.001)## GeV

process.LHCTransport.Hector.smearAng    = cms.bool(True)
#process.LHCTransport.Hector.sigmaSTX    = cms.double(0.01)## urad
#process.LHCTransport.Hector.sigmaSTY    = cms.double(0.01)## urad

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
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/z/zhokin/fff/FP420development/data/ExhumeHbb/V384/ExHuME_CEPHiggs200Tobb_14TeV_cff_py_GEN.root')
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
    fileName = cms.untracked.string('HecFP420_test.root')
                              )
#    fileName = cms.untracked.string('HecFP420_Hbb200_14TeV_vtx0HecSmear.root')
#    fileName = cms.untracked.string('HecExhume_Hbb_14TeV_20ev.root')

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
##   for VtxSmearedGauss:
process.VtxSmeared.MeanX = cms.double(0.0)
process.VtxSmeared.MeanY = cms.double(0.0)
process.VtxSmeared.MeanZ = cms.double(0.01)
process.VtxSmeared.SigmaX = cms.double(0.005)
process.VtxSmeared.SigmaY = cms.double(0.005)
process.VtxSmeared.SigmaZ = cms.double(9.0)
#
#process.FP420Digi.FedFP420Algorithm = 2 # was 1
process.FP420Digi.FedFP420LowThreshold =4. # was 6.
process.FP420Digi.FedFP420HighThreshold =4.5 # was 6.5
#process.FP420Digi.NoFP420Noise = True # was False
process.FP420Digi.AdcFP420Threshold =5.0 # was 6.0
#process.FP420Digi.AddNoisyPixels = True # was True
#process.FP420Digi.ApplyTofCut = True   # was True
#process.FP420Digi.VerbosityLevel = -50
#process.FP420Cluster.VerbosityLevel = 1
process.FP420Cluster.ChannelFP420Threshold =5.0 # was 8.
process.FP420Cluster.ClusterFP420Threshold =6.0 # was 9.
process.FP420Cluster.SeedFP420Threshold =5.5 # was 8.5
#process.FP420Cluster.MaxVoidsFP420InCluster = 1 # was 1
#process.FP420Track.VerbosityLevel = -29
#process.FP420Track.VerbosityLevel = -30
#process.FP420Track.VerbosityLevel = 1
process.FP420Track.chiCutY420 = 3. # means chi2Y (was=2)
process.FP420Track.chiCutX420 = 3. # means chi2Y (was=50)
#process.FP420Reco.VerbosityLevel = -49
#process.FP420Reco.VtxFlagGenRec = 1



