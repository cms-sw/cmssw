# The following comments couldn't be translated into the new config version:

# Configuration file to run CSCGeometryOfWires
# to dump wire & strip info from CSC geometry.
# Tim Cox 16.09.2008

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# xml for endcap csc geometry
# ===========================
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# Fake alignment is/should be ideal geometry
process.load("CalibMuon.Configuration.Muon_FakeAlignment_cff")

# flags for modelling of CSC layer & strip geometry
# =================================================
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        CSCWireTopology = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        noLineBreaks = cms.untracked.bool(True),
## DEBUG will dump addresses of CSCChamberSpecs objects etc. INFO does not.        
##        threshold = cms.untracked.string('DEBUG'),
        threshold = cms.untracked.string('INFO'),
        CSCChamberSpecs = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    categories = cms.untracked.vstring('CSC', 
        'CSCChamberSpecs', 
        'CSCWireTopology'),
    destinations = cms.untracked.vstring('cout')
)

process.producer = cms.EDFilter("CSCGeometryOfWires")

process.p1 = cms.Path(process.producer)

