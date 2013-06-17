import FWCore.ParameterSet.Config as cms

#########################################################################################################################
#
# Example to show how to run the real tracking instead of the emulated one after having created the tracker hits
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

# Common inputs, with fake conditions (not fake ay more!)
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('FastSimulation.Configuration.Geometries_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
process.load("FastSimulation/Configuration/FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
#process.load('FastSimulation.PileUpProducer.PileUpSimulator_2012_Startup_inTimeOnly_cff')
process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
# You may not want to simulate everything for your study
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True

# Needed to run the tracker digitizers
process.load('Configuration.StandardSequences.Digi_cff')
process.load('SimGeneral.MixingModule.pixelDigitizer_cfi')
process.pixelDigitizer.hitsProducer =  'famosSimHitsTrackerHits'
process.pixelDigitizer.makeDigiSimLinks = False # if you set to True, you need some more replacements
process.load('SimGeneral.MixingModule.stripDigitizer_cfi')
process.stripDigitizer.hitsProducer =  'famosSimHitsTrackerHits'
process.stripDigitizer.ROUList = ['famosSimHitsTrackerHits']
process.load('SimTracker.SiStripDigitizer.SiStripDigiSimLink_cfi')
process.simSiStripDigiSimLink.ROUList = ['famosSimHitsTrackerHits']

# Needed to run the tracker local reco
#process.load('RecoTracker.Configuration.RecoTracker_cff')

# Produce Tracks and Clusters
process.source = cms.Source("EmptySource")
process.p1 = cms.Path(process.generator*process.famosWithTracksAndMuonHits) # choose any sequence that you like in FamosSequences_cff
process.p2 = cms.Path(process.trDigi) # real digitizers
#process.reconstruction_step     = cms.Path(process.trackerlocalreco)

# To write out events (not need: FastSimulation _is_ fast!)
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("MyFirstFamosFile_1.root"),
    outputCommands = cms.untracked.vstring("keep *",
                                           "drop *_mix_*_*")
    )

process.outpath = cms.EndPath(process.o1)

#process.schedule = cms.Schedule(process.p1,process.p2,process.reconstruction_step,process.outpath)
process.schedule = cms.Schedule(process.p1,process.p2,process.outpath)

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
