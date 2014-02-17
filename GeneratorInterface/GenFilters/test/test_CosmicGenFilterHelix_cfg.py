#
# Original Author:  Gero Flucke
#         Created:  15.8.2008
#             $Id: test_CosmicGenFilterHelix_cfg.py,v 1.1 2008/08/15 20:25:04 flucke Exp $
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
#process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                   sourceSeed = cms.untracked.uint32(8913579)
                                                   )
# message logger 
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    limit = cms.untracked.int32(10)
    ))
process.MessageLogger.statistics.append('cout')

# magnetic field - in principle easy, but...
process.load("Configuration.StandardSequences.MagneticField_cff")
# ... difficult for 0 T since need to add all these lines
# # (cf. Configuration/GlobalRuns/python/ForceZeroTesla_cff.py
# #  and Configuration/GlobalRuns/python/recoT0DQM_EvContent_cfg.py):
# process.localUniform = cms.ESProducer("UniformMagneticFieldESProducer",
#                                       ZFieldInTesla = cms.double(0.0)
#                                       )
# process.es_prefer_localUniform = cms.ESPrefer("UniformMagneticFieldESProducer","localUniform")
# # We use the SteppingHelixPropagatorAlong in filter and have to prepare for 0 T as well, sigh:
# process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
# process.SteppingHelixPropagatorAlong.useInTeslaFromMagField = True
# process.SteppingHelixPropagatorAlong.SetVBFPointer = True
# process.VolumeBasedMagneticFieldESProducer.label = 'VolumeBasedMagneticField'

# geometry
process.load("Configuration.StandardSequences.Geometry_cff")

# source
process.load("GeneratorInterface.CosmicMuonGenerator.CMSCGENsource_cfi")
process.source.MinP = 5.
#process.source.MaxTheta = 70. # just to speed up - but angular bias...

# filter with histogram service
process.load("GeneratorInterface.GenFilters.CosmicGenFilterHelix_cff")
process.cosmicInTracker.doMonitor = True # needs TFileService:
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('test_CosmicGenFilterHelix.root')
)

# output
process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(1000), take care, cut on output might lead to long jobs...
    output = cms.untracked.PSet(FEVT = cms.untracked.int32(100))
    )
process.FEVT = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('genCosmicsReachTracker.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('filterPath')
    )
)

# paths
process.filterPath = cms.Path(process.cosmicInTracker)
process.outputPath = cms.EndPath(process.FEVT)

