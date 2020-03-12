# The following comments couldn't be translated into the new config version:

# Configuration file for EventSetupTest_t

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
# CondDB
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# Geometry
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

#process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
#process.load("Geometry.DTGeometryBuilder.dtGeometry_cfi")

process.DTGeometryESModule.applyAlignment = cms.bool(False)

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

