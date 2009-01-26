# Configuration file to run CSCGeometryOfWires
# to dump wire & strip info from CSC geometry.
# Tim Cox 21.01.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryOfWires")

# Endcap Muon geometry
# ====================
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonEndcapIdealGeometryXML_cfi")

# Fake alignment is/should be ideal geometry
# ==========================================
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.fake2 = process.FakeAlignmentSource
del process.FakeAlignmentSource
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "fake2")


# flags for modelling of CSC layer & strip geometry
# =================================================
process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    ## DEBUG will dump addresses of CSCChamberSpecs objects etc. INFO does not.        
    threshold = cms.untracked.string('INFO'),
    categories = cms.untracked.vstring(
       'CSC', 
       'CSCChamberSpecs', 
       'CSCWireTopology'
    ),
    destinations = cms.untracked.vstring('cout'),
    noLineBreaks = cms.untracked.bool(True),                                    
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
      CSCChamberSpecs = cms.untracked.PSet(
         limit = cms.untracked.int32(-1)
      )
   )
)

process.producer = cms.EDAnalyzer("CSCGeometryOfWires")

process.p1 = cms.Path(process.producer)

