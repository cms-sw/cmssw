import FWCore.ParameterSet.Config as cms

hltEgammaSuperClustersToPixelMatchL1Seeded = cms.EDProducer("EgammaHLTFilteredSuperClusterProducer",
    cands = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    cuts = cms.VPSet(cms.PSet(
        barrelCut = cms.PSet(
            cutOverE = cms.double(0.2),
            useEt = cms.bool(False)
        ),
        endcapCut = cms.PSet(
            cutOverE = cms.double(0.2),
            useEt = cms.bool(False)
        ),
        var = cms.InputTag("hltEgammaHoverEL1Seeded")
    )),
    minEtCutEB = cms.double(10.0),
    minEtCutEE = cms.double(10.0)
)
