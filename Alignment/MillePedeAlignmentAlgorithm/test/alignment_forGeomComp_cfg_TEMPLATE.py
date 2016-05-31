# Config file template to produce a treeFile.root usable as input for
# .../CMSSW/Alignment/MillePedeAlignmentAlgorithm/macros/CompareMillePede.h
# to compare two different geometries.
#
# to be replaced using e.g. sed -e "s/GLOBALTAG/FT_R_53_V6C/h" inFile > outFile:
#
#GLOBALTAG  # without '::All'
#RUNNUMBER
#TREEFILE
#LOGFILE

# last update on $Date: 2011/10/20 16:37:13 $ by $Author: flucke $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # do not accept this exception
    )

# initialize  MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.LOGFILE = cms.untracked.PSet(
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
        )
    )
process.MessageLogger.cerr.placeholder = cms.untracked.bool(True)
process.MessageLogger.destinations = ['LOGFILE']
process.MessageLogger.statistics = ['LOGFILE']
process.MessageLogger.categories = ['Alignment']


# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
# geometry
process.load("Configuration.Geometry.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.Geometry_cff")
# global tag and other conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
# take your favourite global tag
process.GlobalTag.globaltag = 'GLOBALTAG'
## if alignment constants not from global tag, add this
from CondCore.DBCommon.CondDBSetup_cfi import *
## CondDBSetup.DBParameters.authenticationPath = '...' # needed to access cms_orcoff_prod
#process.trackerAlignment = cms.ESSource(
#    "PoolDBESSource",
#    CondDBSetup,
##    connect = cms.string("sqlite_file:TrackerAlignment_GR10v6_offline_append.db"),
#    connect = cms.string("frontier://FrontierProd/CMS_COND_31X_ALIGNMENT"),
##    connect = cms.string("oracle://cms_orcoff_prod/CMS_COND_31X_ALIGNMENT"),
#    toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
#                               tag = cms.string("TrackerAlignment_GR10_v6_offline")
##                               tag = cms.string("Alignments")
##                               ),
##                      cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
##                               tag = cms.string("TrackerIdealGeometryErrors210_mc")
#                               )
#                      )
#    )
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")


# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

process.AlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
        'TrackerTPBModule,111111',
        'TrackerTPEModule,111111',
#        'TrackerTIBModuleUnit,101111',
#        'TrackerTIDModuleUnit,101111',
        'TrackerTOBModuleUnit,101111',
#        'TrackerTECModuleUnit,101111'
        )
    )

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

process.AlignmentProducer.saveToDB = True # should not be needed, but is:
#                     otherwise AlignmentProducer does not  call relevant algorithm part
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:remove_me.db'),
    toPut = cms.VPSet(cms.PSet(
      record = cms.string('TrackerAlignmentRcd'),
      tag = cms.string('dummyTagAlignment')
      )
                      )
    )
