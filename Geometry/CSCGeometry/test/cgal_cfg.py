# Configuration file to run CSCGeometryAsLayers
# printing table of layer information.
# Tim Cox 21.01.2009

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryAsLayers")

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

# Care! The following MessageLogger config deactivates even error messges
# from other modules. Try removing altogether to see any!
process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    ## DEBUG will dump addresses of CSCChamberSpecs objects etc. INFO does not.        
    threshold = cms.untracked.string('INFO'),
    categories = cms.untracked.vstring(
       'CSC'
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
      CSC = cms.untracked.PSet(
         limit = cms.untracked.int32(-1)
      )
   )
)

process.producer = cms.EDAnalyzer("CSCGeometryAsLayers")

process.p1 = cms.Path(process.producer)

