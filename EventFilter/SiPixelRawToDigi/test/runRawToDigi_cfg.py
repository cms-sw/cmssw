
import FWCore.ParameterSet.Config as cms

process = cms.Process("RawToDigiTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/0089689A-0A9E-DD11-ABC3-001D09F2512C.root')
)


process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Cabling
#  include "CalibTracker/Configuration/data/Tracker_FakeConditions.cff"
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

#process.siPixelDigis = cms.EDAnalyzer("SiPixelRawDumper",
#    Timing = cms.untracked.bool(False),
#    IncludeErrors = cms.untracked.bool(True),
#    InputLabel = cms.untracked.string('source'),
#    CheckPixelOrder = cms.untracked.bool(False)
#)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:/scratch/dkotlins/digis.root'),
    outputCommands = cms.untracked.vstring("drop *","keep *_siPixelDigis_*_*")
#    untracked vstring outputCommands = { "drop *", "keep *_siPixelDigis_*_*"}
)


# process.s = cms.Sequence(process.dumper)

process.p = cms.Path(process.siPixelDigis)

process.ep = cms.EndPath(process.out)


