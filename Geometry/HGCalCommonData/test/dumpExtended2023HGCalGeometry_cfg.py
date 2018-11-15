import FWCore.ParameterSet.Config as cms

process = cms.Process("GEODUMP")
process.load("Geometry.HGCalCommonData.testGeometryExtended_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('G4cerr')
    process.MessageLogger.categories.append('G4cout')
    process.MessageLogger.categories.append('HGCalGeom')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.add_(cms.ESProducer("TGeoMgrFromDdd",
                            verbose = cms.untracked.bool(False),
                            level = cms.untracked.int32(14)
                            ))

process.dump = cms.EDAnalyzer("DumpSimGeometry",
                              outputFileName = cms.untracked.string('CMS2023D28.root'))

process.p = cms.Path(process.dump)
