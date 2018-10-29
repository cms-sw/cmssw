import FWCore.ParameterSet.Config as cms

hiCentrality = cms.EDProducer("CentralityProducer",

                            produceHFhits = cms.bool(True),
                            produceHFtowers = cms.bool(True),
                            produceEcalhits = cms.bool(True),
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
                            srcZDChits = cms.InputTag("zdcreco"),
                            srcPixelhits = cms.InputTag("siPixelRecHits"),
                            srcTracks = cms.InputTag("hiGeneralTracks"),
                            srcVertex= cms.InputTag("hiSelectedVertex"),
                            srcReUse = cms.InputTag("hiCentrality"),
                            srcPixelTracks = cms.InputTag("hiPixel3PrimTracks"),

                            doPixelCut = cms.bool(True),
                            UseQuality = cms.bool(True),
                            TrackQuality = cms.string('highPurity'),
                            trackEtaCut = cms.double(2),
                            trackPtCut = cms.double(1),
                            hfEtaCut = cms.double(4), #hf above the absolute value of this cut is used
                            midRapidityRange = cms.double(1),
                           lowGainZDC = cms.bool(True),

                            )

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(hiCentrality,
               producePixelTracks = True,
               srcPixelTracks = "hiConformalPixelTracks",
               srcTracks = cms.InputTag("generalTracks"),
               srcVertex = cms.InputTag("offlinePrimaryVertices")
               )

