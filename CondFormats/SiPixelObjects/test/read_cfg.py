import FWCore.ParameterSet.Config as cms

process = cms.Process("ReadCablingTest")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.load("CalibTracker.Configuration.SiPixelCabling.SiPixelCabling_SQLite_cff")
process.siPixelCabling.connect = 'sqlite_file:cabling.db'
process.siPixelCabling.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelFedCablingMapRcd'),
    tag = cms.string('SiPixelFedCablingMap_v16')
))

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        read = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.readstruct =  cms.EDAnalyzer("SiPixelFedCablingMapAnalyzer")

process.p = cms.Path(process.readstruct)

