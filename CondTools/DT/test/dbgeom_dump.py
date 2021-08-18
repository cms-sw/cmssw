# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# CondDB
process.load("CondCore.CondDB.CondDB_cfi")

# Geometry
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('Configuration.StandardSequences.DD4hep_GeometrySim_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('myLogDT'),
    myLogDT = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
    )
)

process.muonGeometryConstants.fromDD4Hep = True
process.DTGeometryESModule.applyAlignment = cms.bool(False)
process.DTGeometryESModule.fromDDD = cms.bool(True)
process.DTGeometryESModule.fromDD4hep = cms.bool(False)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(54100)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dump = cms.EDAnalyzer("DTGeometryDump"
)

process.p = cms.Path(process.dump)

