import FWCore.ParameterSet.Config as cms

process = cms.Process("LAS")

##
## Message Logger
##
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100) # every 100th only
#    limit = cms.untracked.int32(10)       # or limit to 10 printouts...
    ))
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

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
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_3_2_4/src/Alignment/LaserAlignment/test/tkLasBeams_ideal_newtest.root' # tkLasBeams_noATs_CRAFT08.root # tkLasBeams_CRAFT09.root # tkLasBeams_Run123353_39.root # tkLasBeams_Craft09_goodHits.root
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
process.GlobalTag.globaltag = "DESIGN_31X_V5::All" # DESIGN_31X_V5::All # MC_31X_V5::All # CRAFT09_R2_V2::All # CRAFT0831X_V4::All

# output file
process.TFileService = cms.Service("TFileService", 
                                   fileName = cms.string("histograms.root"),
                                   closeFileFast = cms.untracked.bool(True)
)


# using database file

#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
#                                        connect = cms.string("frontier://FrontierProd/CMS_COND_31X_FROM21X"),
#                                        connect = cms.string("sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/alignmentObjects/kaschube/LAS_CRAFT08_fixDisks19.db"),
#                                        connect = cms.string("sqlite_file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_3_2_4/src/Alignment/MillePedeAlignmentAlgorithm/test/LAS_Ideal_Gaussian.db"),
#                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
#                                                                   tag = cms.string("Alignments"))#, #"Alignments"
#                                                          cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
#                                                                   tag = cms.string("")) #"AlignmentErrorsExtended"
#                                        )
#)
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")

##
## My module(s)
##
process.load("Alignment.LaserAlignment.TkLasBeamFitter_cfi")
process.TkLasBeamFitter.fitBeamSplitters = True
process.TkLasBeamFitter.numberOfFittedAtParameters = 6 # '3' or '5', default is '6'

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



