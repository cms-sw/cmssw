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
    fileNames = cms.untracked.vstring("file:/afs/cern.ch/user/f/flucke/cms/CMSSW/CMSSW_2_2_8/tkLasBeams.root"
                                      )
    )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )


##
## Geometry and conditions (add B-field?)
##

# too many things...process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_V12::All"

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



