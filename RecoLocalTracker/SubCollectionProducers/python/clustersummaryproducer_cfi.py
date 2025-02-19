import FWCore.ParameterSet.Config as cms

clusterSummaryProducer = cms.EDProducer('ClusterSummaryProducer',
                                        stripClusters=cms.InputTag('siStripClusters'),
                                        pixelClusters=cms.InputTag('siPixelClusters'),
                                        stripModule=cms.string('TOB,TIB,TID,TEC,TRACKER'),
                                        stripVariables=cms.string('cHits,cSize,cCharge'),
                                        pixelModule=cms.string('BPIX,FPIX,PIXEL'),
                                        pixelVariables=cms.string('pHits,pSize,pCharge'),
                                        doStrips=cms.bool(True),
                                        doPixels=cms.bool(True),
                                        verbose=cms.bool(False)
                                        )
