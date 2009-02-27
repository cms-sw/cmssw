import FWCore.ParameterSet.Config as cms    

process = cms.Process("USER")
process.add_(cms.Service( "MessageLogger"))
process.source = cms.Source( "EmptySource" )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )

process.load("PerfTools.Callgrind.callgrindSwitch_cfi")

process.SiStripNoisesRcdSource = cms.ESSource( "EmptyESSource",
                                               recordName = cms.string( "SiStripNoisesRcd" ),
                                               iovIsRunNotTime = cms.bool( True ),
                                               firstValid = cms.vuint32( 1 )
                                               )
process.SiStripGainRcdSource = cms.ESSource( "EmptyESSource",
                                             recordName = cms.string( "SiStripGainRcd" ),
                                             iovIsRunNotTime = cms.bool( True ),
                                             firstValid = cms.vuint32( 1 )
                                             )
process.SiStripQualityRcdSource = cms.ESSource( "EmptyESSource",
                                                recordName = cms.string( "SiStripQualityRcd" ),
                                                iovIsRunNotTime = cms.bool( True ),
                                                firstValid = cms.vuint32( 1 )
                                                )

process.load("RecoLocalTracker.SiStripClusterizer.test.ClusterizerUnitTestFunctions_cff")
process.load("RecoLocalTracker.SiStripClusterizer.test.ClusterizerUnitTests_cff")
testDefinition = cms.VPSet() + [process.ClusterizerDefaultGroup,process.ClusterizerProposedGroup]

process.es           = cms.ESProducer("ClusterizerUnitTesterESProducer",
                                      ClusterizerTestGroups = testDefinition   )
process.runUnitTests = cms.EDAnalyzer("ClusterizerUnitTester",
                                      ClusterizerTestGroups = testDefinition  )

process.path = cms.Path( process.profilerStart * process.runUnitTests * process.profilerStop)
