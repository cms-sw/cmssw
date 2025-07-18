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
                            producePF = cms.bool(True),
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
                            srcPF = cms.InputTag("particleFlow"),

                            doPixelCut = cms.bool(True),
                            useQuality = cms.bool(True),
                            trackQuality = cms.string('highPurity'),
                            trackEtaCut = cms.double(2),
                            trackPtCut = cms.double(1),
                            hfEtaCut = cms.double(4), #hf above the absolute value of this cut is used
                            midRapidityRange = cms.double(1),
                            lowGainZDC = cms.bool(True),
                            isPhase2 = cms.bool(False),

                            )

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
(pp_on_XeXe_2017 | pp_on_AA | run3_upc).toModify(hiCentrality,
                                      producePixelTracks = True,
                                      srcPixelTracks = "hiConformalPixelTracks",
                                      srcTracks = "generalTracks",
                                      srcVertex = "offlinePrimaryVertices"
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(hiCentrality, srcZDChits = "zdcrecoRun3",lowGainZDC = False)

from Configuration.ProcessModifiers.phase2_pp_on_AA_cff import phase2_pp_on_AA
phase2_pp_on_AA.toModify(hiCentrality,
    isPhase2 = True,
    producePixelTracks = False,
    srcTracks = "generalTracks",
    srcVertex = "offlinePrimaryVertices"
)