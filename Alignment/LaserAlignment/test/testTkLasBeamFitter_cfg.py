import FWCore.ParameterSet.Config as cms

process = cms.Process("LAS")

##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))
process.MessageLogger.statistics.append('cout')

##
## Process options
##
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # make this exception fatal
)

##
## Data input
##
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_2_2_10/src/Alignment/LaserAlignment/tkLasBeams_dataCRAFT.root' # tkLasBeams_dataCRAFT.root, tkLasBeams_CRAFT_2_2_9.root
                                      )
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## Geometry and conditions
##
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff") # for 22X when using db object
process.GlobalTag.globaltag = "IDEAL_V12::All"

# using database file

from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
#                                        connect = cms.string("sqlite_file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_2_2_10/src/LasReader/TestProducer/alignments_MP.db"),
                                        connect = cms.string("frontier://FrontierProd/CMS_COND_21X_ALIGNMENT"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("Tracker_Geometry_v5_offline")), #"Alignments"
                                                          cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
                                                                   tag = cms.string("Tracker_GeometryErr_v5_offline")) #"AlignmentErrors"
                                       )
)
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

##
## My module(s)
##
process.load("Alignment.LaserAlignment.TkLasBeamFitter_cfi")

process.out = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 'keep Tk*Beams_*_*_*'),
    fileName = cms.untracked.string('./tkFittedLasBeams.root')
    )


##
## paths
##
process.path       = cms.Path(process.TkLasBeamFitter)
process.outputPath = cms.EndPath(process.out)



