import FWCore.ParameterSet.Config as cms

process = cms.Process("MFGEOMDUMP")
process.source = cms.Source("EmptySource")
process.load("MagneticField.GeomBuilder.cmsMFGeometryXML_160812_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(14)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry", 
                              tag = cms.untracked.string("MFGeometry"),
                              outputFileName = cms.untracked.string("cmsDDMFGeometryTest.root"))

process.p = cms.Path(process.dump)
