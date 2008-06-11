# The following comments couldn't be translated into the new config version:

# Configuration file to run CSCGeometryOfStrips
# to check some strip info from CSC geometry.
# Tim Cox 01.05.2007

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# xml for endcap csc geometry
# ===========================
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# flags for modelling of CSC layer & strip geometry
# =================================================
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    #      untracked vstring debugModules   = { "*" }
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CSCWireTopology = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG'),
        CSCChamberSpecs = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('CSC', 
        'CSCChamberSpecs', 
        'CSCWireTopology'),
    destinations = cms.untracked.vstring('cout')
)

process.producer = cms.EDFilter("CSCGeometryOfStrips")

process.p1 = cms.Path(process.producer)

