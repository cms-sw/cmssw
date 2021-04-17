import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# DT geometry 
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

# CSC geometry
#include "Geometry/MuonCommonData/data/muonEndcapIdealGeometryXML.cfi"
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# Misalignment example scenario producer
process.load("Alignment.MuonAlignment.Scenarios_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    files = cms.untracked.PSet(
        info_txt = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.MisalignedTracker = cms.ESProducer("MisalignedTrackerESProducer",
    process.MuonNoMovementsScenario
)

process.myprod = cms.EDAnalyzer("TestTranslation",
    fileName = cms.untracked.string('misaligned-2.root')
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.myprod)
process.ep = cms.EndPath(process.asciiPrint)


