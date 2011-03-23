import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    
# For valgrind studies
# process.ProfilerService = cms.Service("ProfilerService",
#    lastEvent = cms.untracked.int32(13),
#    firstEvent = cms.untracked.int32(3),
#    paths = cms.untracked.vstring('p1')
#)

# Include the RandomNumberGeneratorService definition
process.load("FastSimulation/Configuration/RandomServiceInitialization_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'file:NeutrinoFullSimNoZS.root'),
                            noEventSort = cms.untracked.bool(True),
                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
                            )



# To make histograms
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']


# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.noiseCheck = cms.EDAnalyzer("NoiseCheck",
                                  OutputFile=cms.string('Noisecheck-Neutrino-fast-final.root'),
                                  Threshold=cms.double(0.318))

# Produce Tracks and Clusters
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithCaloHits*process.noiseCheck)
process.p1 = cms.Path(process.noiseCheck)


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
