# Config file template to produce a treeFile.root usable as input for
# .../CMSSW/Alignment/MillePedeAlignmentAlgorithm/macros/CompareMillePede.h
# to compare two different geometries.
#
#GLOBALTAG  # without '::All'
#RUNNUMBER
#TREEFILE
#LOGFILE

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # do not accept this exception
    )

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
# geometry
process.load("Configuration.Geometry.GeometryRecoDB_cff")
# global tag and other conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# take your favourite global tag
from Configuration.AlCa.GlobalTag import GlobalTag 
process.GlobalTag = GlobalTag(process.GlobalTag, "GLOBALTAG")     
usedGlobalTag = process.GlobalTag.globaltag.value()

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.files.LOGFILE = cms.untracked.PSet(
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(5),
        reportEvery = cms.untracked.int32(5)
        ),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(10)
        ),
    ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    Alignment = cms.untracked.PSet(
        limit = cms.untracked.int32(-1),
        reportEvery = cms.untracked.int32(1)
        ),
    enableStatistics = cms.untracked.bool(True)
    )
process.MessageLogger.cerr.enable = cms.untracked.bool(False)


## if alignment constants not from global tag, add this
from CondCore.CondDB.CondDB_cfi import *
CondDBReference = CondDB.clone(connect = cms.string('sqlite_file:remove_me.db'))

# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

process.AlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        "PixelHalfBarrelDets,111111",
        "PXECDets,111111",
        #  'TrackerTIBModuleUnit,101111',
        #  'TrackerTIDModuleUnit,101111',
        'TrackerTOBModuleUnit,101111',
        #  'TrackerTECModuleUnit,101111'
        )
    )

# explicitely specify run ranges to convince algorithm that multi-IOV input is fine
process.AlignmentProducer.RunRangeSelection = [
    cms.PSet(
        RunRanges = cms.vstring("RUNNUMBER"),
        selector = process.AlignmentProducer.ParameterBuilder.Selector.alignParams
    )
] # end of process.AlignmentProducer.RunRangeSelection

# enable alignable updates to convince algorithm that multi-IOV input is fine
process.AlignmentProducer.enableAlignableUpdates = True

process.AlignmentProducer.doMisalignmentScenario = False #True
process.AlignmentProducer.applyDbAlignment = True # either globalTag or trackerAlignment
process.AlignmentProducer.checkDbAlignmentValidity = False

# assign by reference (i.e. could change MillePedeAlignmentAlgorithm as well):
process.AlignmentProducer.algoConfig = process.MillePedeAlignmentAlgorithm

process.AlignmentProducer.algoConfig.mode = 'pedeRead'
process.AlignmentProducer.algoConfig.treeFile = 'TREEFILE'
process.AlignmentProducer.algoConfig.pedeReader.readFile = 'FILE_MUST_NOT_EXIST.res'

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(RUNNUMBER) # choose your run!
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1) )


process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.dump)

process.AlignmentProducer.saveToDB = True # should not be needed, but is: otherwise AlignmentProducer does not  call relevant algorithm part
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBReference,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                     tag = cms.string('dummyTagAlignment')
                                                                     )
                                                            )
                                          )
