import FWCore.ParameterSet.Config as cms

process = cms.Process("ClusterSummaryProducer")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",                            
                            fileNames = cms.untracked.vstring(
                            #'/store/data/Run2010B/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/146/511/A87F9318-92C7-DF11-985E-001D09F27003.root'
                             '/store/data/Run2011A/MinimumBias/ALCARECO/SiStripCalMinBias-v4/000/165/548/8449E5BF-8C87-E011-9F5D-003048F118C2.root'
                            )
                            )

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)

process.ClusterSummaryProducer = cms.EDProducer('ClusterSummaryProducer',
                                        stripClusters=cms.InputTag("siStripClusters"),
                                        pixelClusters=cms.InputTag("siPixelClusters"),
                                        #Module=cms.string('TOB, TIB, TID, TEC, TIB_1, TOB_4, TECM, TECP, TIDM, TIDP, TECM_1, TECP_2, TIDM_3, TIDP_1'),
                                        #Module=cms.string( TOB, TIB, TID, TEC, TRACKER')
                                        stripModule=cms.string('TOB,TIB'),        
                                        #Module=cms.string('TECP,TECP_1,TECP_2,TECP_3,TECP_4,TECP_5,TECP_6,TECP_7,TECP_8,TECP_9'),
                                        #Module=cms.string('TOB, TECPR_2, TIDMR_2'),
                                        stripVariables=cms.string('cHits,cSize,cCharge'),
                                        #pixelModule=cms.string('BPIX,FPIX,FPIXM,FPIXP,PIXEL'),        
                                        pixelModule=cms.string('BPIX,FPIX'),        
                                        pixelVariables=cms.string('pHits,pSize,pCharge'),
                                        doStrips=cms.bool(True),
                                        doPixels=cms.bool(True),
                                        verbose=cms.bool(False)
                                        )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('myOutputFile.root')
                               ,outputCommands = cms.untracked.vstring('drop *',
                                                                       'keep *_ClusterSummaryProducer_*_*',
                                                                       )
                               )


process.p = cms.Path(process.ClusterSummaryProducer)

process.e = cms.EndPath(process.out)
