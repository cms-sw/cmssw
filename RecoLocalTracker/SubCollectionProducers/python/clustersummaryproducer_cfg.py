import FWCore.ParameterSet.Config as cms

process = cms.Process("ClusterSummaryProducer")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source("PoolSource",                            
                            fileNames = cms.untracked.vstring(
                             '/store/data/Run2011A/MinimumBias/ALCARECO/SiStripCalMinBias-v4/000/165/548/8449E5BF-8C87-E011-9F5D-003048F118C2.root'
                            )
                            )

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.clusterSummaryProducer = cms.EDProducer('ClusterSummaryProducer',
                                        stripClusters=cms.InputTag("siStripClusters"),
                                        pixelClusters=cms.InputTag("siPixelClusters"),                
                                        doStrips=cms.bool(True),
                                        doPixels=cms.bool(True),
                                        verbose=cms.bool(False),
                                        wantedSubDets = cms.VPSet(    
                                          cms.PSet(detSelection = cms.uint32(2),detLabel = cms.string("TOB"), selection=cms.vstring("0x1e000000-0x1A000000")),
                                          cms.PSet(detSelection = cms.uint32(1),detLabel = cms.string("TIB"), selection=cms.vstring("0x1e000000-0x16000000")),
                                          cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TID"), selection=cms.vstring("0x1e000000-0x18000000")),
                                          cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TEC"), selection=cms.vstring("0x1e000000-0x1C000000")),
                                          cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TRACKER"), selection=cms.vstring("0x1e000000-0x1A000000",
                                                                                                                                        "0x1e000000-0x16000000",
                                                                                                                                        "0x1e000000-0x18000000",
                                                                                                                                        "0x1e000000-0x1C000000",
                                                                                                                                        )),
                                          cms.PSet(detSelection = cms.uint32(7),detLabel = cms.string("BPIX"),selection=cms.vstring("0x1e000000-0x12000000")),    
                                          cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("FPIX"),selection=cms.vstring("0x1e000000-0x14000000")),
                                          cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("PIXEL"),selection=cms.vstring("0x1e000000-0x12000000",
                                                                                                                                     "0x1e000000-0x14000000"                                                                                                                                     
                                                                                                                                     )),    
                                          )
                                        )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('myOutputFile.root')
                               ,outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_clusterSummaryProducer_*_*',
                                                                       )
                               )


process.p = cms.Path(process.clusterSummaryProducer)

process.e = cms.EndPath(process.out)

