import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(XXX_FIRSTRUN_XXX),
    lastValue  = cms.uint64(XXX_LASTRUN_XXX)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'XXX_GT_XXX::All'
process.prefer("GlobalTag")

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalibValidation.InputFiles          = cms.vstring(
XXX_CALIBTREE_XXX
)



process.SiStripCalibValidation.FirstSetOfConstants = cms.untracked.bool(False)
process.SiStripCalibValidation.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module


process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Validation_Tree.root')  
)

process.p = cms.Path(process.SiStripCalibValidation)
