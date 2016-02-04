import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')
process.add_( cms.Service("Timing"))
process.load("PerfTools.Callgrind.callgrindSwitch_cfi")
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring("HLTClusterizer"),
                                    destinations = cms.untracked.vstring('log.txt'))

# Configuration which varies with source
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_31X_V2::All'

process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('RecoLocalTracker/SiStripZeroSuppression/SiStripZeroSuppression_SimData_cfi')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20))
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EA8E5AF7-576B-DE11-BA98-001D09F24498.root'))


# Configuration which varies depending on what to compare
process.OldClusterizer = cms.EDProducer("SiStripClusterizer",
                                        Clusterizer = cms.PSet( Algorithm = cms.string("OldThreeThresholdAlgorithm"),
                                                                ChannelThreshold = cms.double(2),
                                                                SeedThreshold    = cms.double(3),
                                                                ClusterThreshold = cms.double(5),
                                                                MaxSequentialHoles = cms.uint32(0),
                                                                QualityLabel = cms.string("")),
                                        DigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','ZeroSuppressed'),
                                                                           cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                                                           cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                                                           cms.InputTag('siStripZeroSuppression','ScopeMode')))
process.NewClusterizer = cms.EDProducer("SiStripClusterizer",
                                        Clusterizer = cms.PSet( Algorithm = cms.string("ThreeThresholdAlgorithm"),
                                                                ChannelThreshold = cms.double(2),
                                                                SeedThreshold    = cms.double(3),
                                                                ClusterThreshold = cms.double(5),
                                                                MaxSequentialHoles = cms.uint32(0),
                                                                MaxSequentialBad   = cms.uint32(0),
                                                                MaxAdjacentBad     = cms.uint32(1),
                                                                QualityLabel = cms.string("")),
                                        DigiProducersList = cms.VInputTag( cms.InputTag('siStripDigis','ZeroSuppressed'),
                                                                           cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                                                           cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                                                           cms.InputTag('siStripZeroSuppression','ScopeMode')))
process.NewStripByStrip = cms.EDProducer("StripByStripTestDriver",
                                         Algorithm = cms.string("ThreeThresholdAlgorithm"),
                                         ChannelThreshold = cms.double(2),
                                         SeedThreshold    = cms.double(3),
                                         ClusterThreshold = cms.double(5),
                                         MaxSequentialHoles = cms.uint32(0),
                                         MaxSequentialBad   = cms.uint32(0),
                                         MaxAdjacentBad     = cms.uint32(1),
                                         QualityLabel = cms.string(""),
                                         DigiProducer = cms.InputTag('siStripDigis','ZeroSuppressed'),
                                         HLT = cms.bool(False)
                                         )
process.HLTStripByStrip = cms.EDProducer("StripByStripTestDriver",
                                         ClusterizerAlgorithm = cms.untracked.string('ThreeThreshold'),
                                         ChannelThreshold = cms.untracked.double(2.0),
                                         SeedThreshold = cms.untracked.double(3.0),
                                         MaxHolesInCluster = cms.untracked.uint32(0),
                                         ClusterThreshold = cms.untracked.double(5.0),

                                         DigiProducer = cms.InputTag('siStripDigis','ZeroSuppressed'),
                                         HLT = cms.bool(True)
                                         )

process.CompareOldNew = cms.EDAnalyzer("CompareClusters",
                                       Clusters1 = cms.InputTag('OldClusterizer',''),
                                       Clusters2 = cms.InputTag('NewClusterizer',''),
                                       Digis     = cms.InputTag('siStripDigis','ZeroSuppressed')
                                       )
process.CompareOldHLT = cms.EDAnalyzer("CompareClusters",
                                       Clusters1 = cms.InputTag('OldClusterizer',''),
                                       Clusters2 = cms.InputTag('HLTStripByStrip',''),
                                       Digis     = cms.InputTag('siStripDigis','ZeroSuppressed')
                                       )
process.CompareNewNew = cms.EDAnalyzer("CompareClusters",
                                       Clusters1 = cms.InputTag('NewClusterizer',''),
                                       Clusters2 = cms.InputTag('NewStripByStrip',''),
                                       Digis     = cms.InputTag('siStripDigis','ZeroSuppressed')
                                       )

process.p1 = cms.Path(   process.siStripDigis *
                         process.siStripZeroSuppression *
                         #process.profilerStart *
                         process.OldClusterizer *
                         process.NewClusterizer *
                         process.HLTStripByStrip *
                         process.NewStripByStrip *
                         #process.profilerStop *
                         
                         process.CompareOldNew *
                         process.CompareOldHLT *
                         process.CompareNewNew
                         )
