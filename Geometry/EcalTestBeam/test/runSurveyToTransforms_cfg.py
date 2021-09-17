
import FWCore.ParameterSet.Config as cms


process = cms.Process("SurveyToTransforms")

#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cout.enable = cms.untracked.bool(True)
#process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
#process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.EcalTestBeam.idealGeomPlusEE_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource")


process.cga = cms.EDAnalyzer("SurveyToTransforms" )

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('survey.root')
                                   )

process.testendcap = cms.ESProducer( "testEcalEndcapGeometryEP",
                                     applyAlignment = cms.bool(False) )

process.es_prefer_endcap = cms.ESPrefer( "testEcalEndcapGeometryEP", "testendcap" )

process.p1 = cms.Path(process.cga)


