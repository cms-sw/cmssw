import FWCore.ParameterSet.Config as cms

hiCentrality = cms.EDFilter("reco::CentralityProducer",

                            doFilter = cms.bool(False),
                            
                            produceHFhits = cms.bool(True),
                            produceHFtowers = cms.bool(True),
                            produceEcalhits = cms.bool(False),
                            produceBasicClusters = cms.bool(True),
                            produceZDChits = cms.bool(True),
                            produceETmidRapidity = cms.bool(True),
                            producePixelhits = cms.bool(True),
                            produceTracks = cms.bool(True),
                            producePixelTracks = cms.bool(True),
                            trackEtaCut = cms.double(2),
                            trackPtCut = cms.double(1),
                            hfEtaCut = cms.double(4), #hf above the absolute value of this cut is used
                            
                            midRapidityRange = cms.double(1),
                            
                            srcHFhits = cms.InputTag("hfreco"),
                            srcTowers = cms.InputTag("towerMaker"),
                            srcEBhits = cms.InputTag("EcalRecHitsEB"),
                            srcEEhits = cms.InputTag("EcalRecHitsEE"),
                            srcBasicClustersEB = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
                            srcBasicClustersEE = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
                            srcZDChits = cms.InputTag("zdcreco"),
			    lowGainZDC = cms.untracked.bool(False),
                            srcPixelhits = cms.InputTag("siPixelRecHits"),
                            doPixelCut = cms.bool(False),
                            srcTracks = cms.InputTag("generalTracks"),
                            srcVertex= cms.InputTag("offlinePrimaryVertices"),
                            UseQuality = cms.bool(True),
                            TrackQuality = cms.string('highPurity'),
                            srcReUse = cms.InputTag("hiCentrality"),
                            srcPixelTracks = cms.InputTag("pixelTracks")
                              )


