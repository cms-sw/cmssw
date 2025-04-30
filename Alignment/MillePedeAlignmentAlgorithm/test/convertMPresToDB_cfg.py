import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("Alignment", Run3)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # do not accept this exception
    )

######################################################
# initialize  MessageLogger
######################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.files.alignment = cms.untracked.PSet(
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

######################################################
# Design GT (in order to fetch the design geometry)
######################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_design', '')  # take your favourite

######################################################
# Starting alignment of the campaign
######################################################
from CondCore.CondDB.CondDB_cfi import *
CondDBConnection = CondDB.clone( connect = cms.string( 'sqlite_file:alignment_input.db' ) )
process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                        CondDBConnection,
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("TrackerAlignment_PCL_byRun_v2_express_348155")
                                                                   )
                                        ))

process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

######################################################
# Alignment producer
######################################################
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

######################################################
#
# !!! This has to match the alignable selection
#          of the pede configuration !!!
#
######################################################
from align_params_cff import _alignParams
process.AlignmentProducer.ParameterBuilder.Selector = _alignParams

######################################################
# Alignment Producer options
######################################################
process.AlignmentProducer.doMisalignmentScenario = False #True
process.AlignmentProducer.applyDbAlignment = True # either globalTag or trackerAlignment

######################################################
# assign by reference (i.e. could change MillePedeAlignmentAlgorithm as well):
######################################################
process.AlignmentProducer.algoConfig = process.MillePedeAlignmentAlgorithm
process.AlignmentProducer.algoConfig.mode = 'pedeRead'     # reads millepede.res
process.AlignmentProducer.algoConfig.pedeReader.readFile = 'millepede.res'
process.AlignmentProducer.algoConfig.treeFile = 'my_treeFile.root'

######################################################
# Source
######################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1) # choose your run!
                            )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1) )

process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.dump)

# should not be needed, but is:
# otherwise AlignmentProducer does not  call relevant algorithm part
process.AlignmentProducer.saveToDB = True

######################################################
# Output alignment payload from reading file
######################################################
CondDBConnectionOut = CondDB.clone( connect = cms.string( 'sqlite_file:convertedFromResFile.db' ) )
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBConnectionOut,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                                                     tag = cms.string('Alignments')
                                                                     )
                                                            )
                                          )
-- dummy change --
