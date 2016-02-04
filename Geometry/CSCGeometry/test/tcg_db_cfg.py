# Configuration file to run stubs/CSCGeometryAnalyser
# to dump CSC geometry information
# Tim Cox 08.04.2009 to test geometry-in-db for 31X

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBESSource = cms.ESSource(
   "PoolDBESSource",
   process.CondDBSetup,
   loadAll = cms.bool(True),
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   toGet = cms.VPSet(
      cms.PSet(
         record = cms.string('CSCRecoGeometryRcd'), tag = cms.string('CSCRECO_Geometry_Test01')
      ),
      cms.PSet(
         record = cms.string('CSCRecoDigiParametersRcd'), tag = cms.string('CSCRECODIGI_Geometry_Test01')
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
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.fake2 = process.FakeAlignmentSource
del process.FakeAlignmentSource
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource", "fake2")
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service(
   'MessageLogger',
   destinations = cms.untracked.vstring('cout'),
   categories = cms.untracked.vstring(
     'CSC',
     'CSCNumbering',
     'CSCGeometryBuilderFromDDD',
     'CSCGeometryBuilder', 
     'CSCGeometryParsFromDD', 
     'RadialStripTopology'
   ),
   debugModules = cms.untracked.vstring('*'),
   cout = cms.untracked.PSet(
      noLineBreaks = cms.untracked.bool(True),
      threshold = cms.untracked.string('DEBUG'),
      default = cms.untracked.PSet(
         limit = cms.untracked.int32(0) # none
      ),
      CSC = cms.untracked.PSet(
         limit = cms.untracked.int32(-1) # all
      ),
      CSCNumbering = cms.untracked.PSet(
         limit = cms.untracked.int32(-1) # all
      ),
      CSCGeometryBuilderFromDDD = cms.untracked.PSet(
         limit = cms.untracked.int32(-1) # all
      ),
      CSCGeometryBuilder = cms.untracked.PSet(
         limit = cms.untracked.int32(0) # none - attempt to match tcg.py output
      ),
      CSCGeometryParsFromDD = cms.untracked.PSet(
         limit = cms.untracked.int32(0) # none 
      ),
      RadialStripTopology = cms.untracked.PSet(
         limit = cms.untracked.int32(-1) # all
      )
   )
)

process.producer = cms.EDAnalyzer("CSCGeometryAnalyzer")

process.CSCGeometryESModule.debugV = True
process.CSCGeometryESModule.useDDD = False

process.p1 = cms.Path(process.producer)

