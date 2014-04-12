import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# DT geometry 
#  include "Geometry/DTGeometry/data/dtGeometry.cfi"
#
# CSC geometry
#   include "Geometry/CSCGeometry/data/cscGeometry.cfi"
process.load("Alignment.MuonAlignment.MisalignedMuon_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    info_txt = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('info_txt', 
        'cerr')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    process.MuonNoMovementsScenario
)

process.prod = cms.EDAnalyzer("TestTranslation",
    fileName = cms.untracked.string('misaligned-2.root')
)

process.p1 = cms.Path(process.prod)


