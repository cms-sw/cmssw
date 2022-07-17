import FWCore.ParameterSet.Config as cms

hltEgammaEleL1TrkIsoUnseeded = cms.EDProducer("EgammaHLTEleL1TrackIsolProducer",
    ecalCands = cms.InputTag("hltEgammaCandidatesUnseeded"),
    eles = cms.InputTag("hltEgammaGsfElectronsUnseeded"),
    isolCfg = cms.PSet(
        etaBoundaries = cms.vdouble(1.5),
        trkCuts = cms.VPSet(
            cms.PSet(
                maxDR = cms.double(0.3),
                maxDZ = cms.double(0.7),
                minDEta = cms.double(0.003),
                minDR = cms.double(0.01),
                minPt = cms.double(2.0)
            ),
            cms.PSet(
                maxDR = cms.double(0.3),
                maxDZ = cms.double(0.7),
                minDEta = cms.double(0.003),
                minDR = cms.double(0.01),
                minPt = cms.double(2.0)
            )
        ),
        useAbsEta = cms.bool(True)
    ),
    l1Tracks = cms.InputTag("TTTracksFromTrackletEmulation","Level1TTTracks")
)
