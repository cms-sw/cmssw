# The following comments couldn't be translated into the new config version:

#keep the logging output to a nice level

import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("CondTools.SiPixel.SiPixelCalibConfiguration_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1570)
)
process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/v/vasquez/scratch0/CMSSW_2_0_0_pre9/src/CalibTracker/SiPixelGainCalibration/test/calibdigis.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.siPixelErrorsDigisToCalibDigis = cms.EDAnalyzer("SiPixelErrorsDigisToCalibDigis",
    saveFile = cms.untracked.bool(True),
    outputFilename = cms.string('myResults.root'),
    SiPixelProducerLabelTag = cms.InputTag("siPixelCalibDigis")
)

process.p = cms.Path(process.siPixelErrorsDigisToCalibDigis)


