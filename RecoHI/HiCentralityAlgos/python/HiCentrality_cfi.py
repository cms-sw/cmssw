import FWCore.ParameterSet.Config as cms

hiCentrality = cms.EDFilter("reco::CentralityProducer",

                            doFilter = cms.bool(False),
                            
                            produceHFhits = cms.bool(True),
                            produceHFtowers = cms.bool(True),
                            produceEcalhits = cms.bool(True),
                            produceBasicClusters = cms.bool(True),
                            produceZDChits = cms.bool(True),
                            produceETmidRapidity = cms.bool(True),
                            producePixelhits = cms.bool(True),
                            produceTracks = cms.bool(True),
                            producePixelTracks = cms.bool(True),
                            reUseCentrality = cms.bool(False),

                            srcHFhits = cms.InputTag("hfreco"),
                            srcTowers = cms.InputTag("towerMaker"),
                            srcEBhits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                            srcEEhits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                            srcVertex = cms.InputTag("hiSelectedVertex"),
                            srcBasicClustersEB = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
                            srcBasicClustersEE = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
                            srcZDChits = cms.InputTag("zdcreco"),
                            srcPixelhits = cms.InputTag("siPixelRecHits"),
                            srcTracks = cms.InputTag("hiSelectedTracks"),
                            srcReUse = cms.InputTag("hiCentrality"),
                            srcPixelTracks = cms.InputTag("hiPixel3PrimTracks"),
                            
                            doPixelCut = cms.bool(True),
                            pixelBarrelOnly = cms.bool(False),
                            trackEtaCut = cms.double(2),
                            trackPtCut = cms.double(1),
                            midRapidityRange = cms.double(1),
			    hfEtaCut = cms.double(4),
                            
                            UseQuality = cms.bool(True),
                            TrackQuality = cms.string('highPurity')
                            
                            )

pACentrality = cms.EDFilter("reco::CentralityProducer",

                            doFilter = cms.bool(False),
                            
                            produceHFhits = cms.bool(True),
                            produceHFtowers = cms.bool(True),
                            produceEcalhits = cms.bool(True),
                            produceBasicClusters = cms.bool(True),
                            produceZDChits = cms.bool(True),
                            produceETmidRapidity = cms.bool(True),
                            producePixelhits = cms.bool(True),
                            produceTracks = cms.bool(True),
                            producePixelTracks = cms.bool(True),
                            reUseCentrality = cms.bool(False),

                            srcHFhits = cms.InputTag("hfreco"),
                            srcTowers = cms.InputTag("towerMaker"),
                            srcEBhits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                            srcEEhits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                            srcVertex = cms.InputTag("offlinePrimaryVertices"),
                            srcBasicClustersEB = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
                            srcBasicClustersEE = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
                            srcZDChits = cms.InputTag("zdcreco"),
                            srcPixelhits = cms.InputTag("siPixelRecHits"),
                            srcTracks = cms.InputTag("generalTracks"),
                            srcReUse = cms.InputTag("pACentrality"),
                            srcPixelTracks = cms.InputTag("pixelTracks"),
                            
                            doPixelCut = cms.bool(True),
                            pixelBarrelOnly = cms.bool(False),
                            trackEtaCut = cms.double(2),
                            trackPtCut = cms.double(0.4),
                            midRapidityRange = cms.double(1),
			    hfEtaCut = cms.double(4), #hf above the absolute value of this cut is used
                            
                            UseQuality = cms.bool(True),
                            TrackQuality = cms.string('highPurity')
                            
                            )


