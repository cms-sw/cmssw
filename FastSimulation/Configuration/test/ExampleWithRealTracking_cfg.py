import FWCore.ParameterSet.Config as cms

#########################################################################################################################
#
# Example to show how to run the real tracking instead of the emulated one after having created the tracker hits.
# In this example also standalone muons are reconstructed (as in standard FastSim), but no other high-level quantity.
#
#########################################################################################################################

process = cms.Process("PROD")

# Include DQMStore, needed by the famosSimHits
process.DQMStore = cms.Service( "DQMStore")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
    
# For valgrind studies
# process.ProfilerService = cms.Service("ProfilerService",
#    lastEvent = cms.untracked.int32(13),
#    firstEvent = cms.untracked.int32(3),
#    paths = cms.untracked.vstring('p1')
#)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=200GeV/c2
process.load("Configuration.Generator.H200ZZ4L_cfi")

# Common inputs
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('FastSimulation.Configuration.Geometries_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
#process.load('FastSimulation.PileUpProducer.PileUpSimulator_2012_Startup_inTimeOnly_cff')
process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')

# Detector simulation with FastSim
process.load("FastSimulation.EventProducer.FamosSimHits_cff")
process.load("FastSimulation.MuonSimHitProducer.MuonSimHitProducer_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# Muon Digi sequence
process.load("SimMuon.Configuration.SimMuon_cff")
process.simMuonCSCDigis.strips.doCorrelatedNoise = False ## Saves a little bit of time

process.simMuonCSCDigis.InputCollection = 'MuonSimHitsMuonCSCHits'
process.simMuonDTDigis.InputCollection = 'MuonSimHitsMuonDTHits'
process.simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'

# Simulation sequence, including digitizers
process.simulationSequence = cms.Sequence(
    process.offlineBeamSpot+
    process.famosMixing+
    process.famosSimHits+
    process.MuonSimHits+
    process.mix+
    process.muonDigi)


# Extend the MixingModule parameterset that tells which digitizers must be executed
from SimGeneral.MixingModule.aliases_cfi import simEcalUnsuppressedDigis, simHcalUnsuppressedDigis, simSiPixelDigis, simSiStripDigis

from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
process.load("CalibTracker.SiStripESProducers.SiStripGainSimESProducer_cfi") # otherwise it complains that no "SiStripGainSimRcd" record is found 

from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimCalorimetry.EcalSimProducers.ecalDigiParameters_cff import *
simEcalUnsuppressedDigis.hitsProducer = cms.string('famosSimHits')
ecal_digi_parameters.hitsProducer = cms.string('famosSimHits')
ecalDigitizer.hitsProducer = cms.string('famosSimHits')

import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi
hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
hcalSimBlockFastSim.hitsProducer = cms.string('famosSimHits')
hcalDigitizer = cms.PSet(
        hcalSimBlockFastSim,
        accumulatorType = cms.string("HcalDigiProducer"),
        makeDigiSimLinks = cms.untracked.bool(False))

process.mix.digitizers = cms.PSet(pixel = cms.PSet(pixelDigitizer),
                                  strip = cms.PSet(stripDigitizer),
                                  ecal = cms.PSet(ecalDigitizer),
                                  hcal = cms.PSet(hcalDigitizer))

process.mix.digitizers.pixel.hitsProducer = cms.string('famosSimHits')
process.mix.digitizers.pixel.ROUList = cms.vstring('TrackerHits')
process.mix.digitizers.strip.hitsProducer = cms.string('famosSimHits')
process.mix.digitizers.strip.ROUList = cms.vstring('TrackerHits')

# Needed to run the tracker reconstruction
process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')
process.siPixelClusters.src = cms.InputTag("mix")
process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag( cms.InputTag('mix','VirginRaw'),
                                                                     cms.InputTag('mix','ProcessedRaw'),
                                                                     cms.InputTag('mix','ScopeMode'))
process.siStripZeroSuppression.DigisToMergeZS = cms.InputTag('mix','ZeroSuppressed')
process.siStripZeroSuppression.DigisToMergeVR = cms.InputTag('mix','VirginRaw')
process.siStripClusters.DigiProducersList = DigiProducersList = cms.VInputTag(
    cms.InputTag('mix','ZeroSuppressed'),
    cms.InputTag('siStripZeroSuppression','VirginRaw'),
    cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
    cms.InputTag('siStripZeroSuppression','ScopeMode'))

process.load('RecoTracker.Configuration.RecoTracker_cff')
process.load('RecoPixelVertexing.Configuration.RecoPixelVertexing_cff')
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')

# Need these lines to stop some errors about missing siStripDigis collections
process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)

# I am not sure about the next lines; taken from http://cmslxr.fnal.gov/lxr/source/SLHCUpgradeSimulations/Geometry/test/GenRecoFull_Fastsim_Stdgeom_cfg.py?v=CMSSW_4_2_8_SLHCstd2
process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(True)  #FG set True to use errors from templates
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(True)  #FG set True to load the last version of the templates
process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
process.PixelCPEGenericESProducer.DoCosmics = False
## CPE for other steps
process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
process.initialStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.lowPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelPairStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.detachedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.mixedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.pixelLessStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
process.tobTecStepTracks.TTRHBuilder = cms.string('WithTrackAngle')

# These are for the muons
process.load('RecoLocalMuon.Configuration.RecoLocalMuon_cff')
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.rpcRecHits.rpcDigiLabel = 'simMuonRPCDigis'
process.dt1DRecHits.dtDigiLabel = 'simMuonDTDigis'
process.dt1DCosmicRecHits.dtDigiLabel = 'simMuonDTDigis'
process.load('RecoMuon.Configuration.RecoMuon_cff')

# Temporary, for debugging
process.dumpContent = cms.EDAnalyzer('EventContentAnalyzer')

# Produce Tracks and Clusters
process.source = cms.Source("EmptySource")
process.gensim_step = cms.Path(process.generator*process.simulationSequence) 
process.reconstruction_step = cms.Path(process.trackerlocalreco
                                       *process.muonlocalreco
                                       *process.standalonemuontracking
                                       *process.recopixelvertexing
                                       *process.ckftracks_wodEdX)

# To write out events
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("OutputFileWithRealTracks.root"),
    outputCommands = cms.untracked.vstring("keep *",
                                           #"drop *_mix_*_*")
                                           "drop *_simEcalUnsuppressedDigis_*_*",
                                           "drop *_simHcalUnsuppressedDigis_*_*")
    )

process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.gensim_step,process.reconstruction_step,process.outpath)
#process.schedule = cms.Schedule(process.gensim_step,process.outpath)

# Keep the logging output to a nice level #

#process.Timing =  cms.Service("Timing")
#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
