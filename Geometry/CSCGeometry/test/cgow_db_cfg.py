# Configuration file to run CSCGeometryOfWires
# to dump wire & strip info from CSC geometry.
# Tim Cox 21.01.2009
# Version to read from database not DDD directly.

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryOfWires")

# Endcap Muon geometry
# ====================

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBESSource = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   loadAll = cms.bool(True),
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   toGet = cms.VPSet(
      cms.PSet(
         record = cms.string('CSCRecoGeometryRcd'), tag = cms.string('XMLFILE_TEST_01')
      ),
      cms.PSet(
         record = cms.string('CSCRecoDigiParametersRcd'), tag = cms.string('XMLFILE_TEST_02')
      )
   ),
   DBParameters = cms.PSet(
      messageLevel = cms.untracked.int32(9),
      authenticationPath = cms.untracked.string('.')
   ),
   catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
   timetype = cms.string('runnumber'),
   connect = cms.string('sqlite_file:myfile.db')

)

# Muon Numbering - strictly shouldn't need it since info is IN the db, but existing dependencies require it
# ==============
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

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

## process.CSCGeometryESModule.debugV = True
process.CSCGeometryESModule.useDDD = False

process.p1 = cms.Path(process.producer)

