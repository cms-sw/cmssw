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
                                        wantedSubDets = cms.vstring("TOB","TIB","TID","TEC","STRIP","BPIX","FPIX","PIXEL"),
                                        wantedUserSubDets = cms.VPSet()
                                        )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('myOutputFile.root')
                               ,outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_clusterSummaryProducer_*_*',
                                                                       )
                               )


process.p = cms.Path(process.clusterSummaryProducer)

process.e = cms.EndPath(process.out)

