import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep

process = cms.Process('DUMP',Run3_dd4hep)

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CSCGeometryBuilder.cscGeometry_cfi")
process.load("Geometry.CSCGeometryBuilder.cscGeometryDump_cfi")

process.CSCGeometryESModule.applyAlignment = False
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.CSCNumberingScheme=dict()
    process.MessageLogger.CSCGeometry=dict()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service(
        "MessageLogger",
  destinations = cms.untracked.vstring(
                'cout'
        ),
        cout = cms.untracked.PSet(
                threshold = cms.untracked.string('DEBUG')
        ),
        debugModules = cms.untracked.vstring('*')
)

process.cscGeometryDump.verbose = True

process.p = cms.Path(process.cscGeometryDump)
