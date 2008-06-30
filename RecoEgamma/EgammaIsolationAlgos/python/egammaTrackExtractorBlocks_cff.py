import FWCore.ParameterSet.Config as cms

EgammaIsoTrackExtractorBlock = cms.PSet(
    ComponentName = cms.string('EcalTrackExtractor'),
    inputTrackCollection = cms.InputTag("generalTracks"),
    DepositLabel = cms.untracked.string(''),
    minCandEt = cms.double(10.),
    Diff_r = cms.double(0.1),
    Diff_z = cms.double(0.2),
    DR_Max = cms.double(0.4),
    DR_Veto = cms.double(0.0),

    BeamlineOption = cms.string('BeamSpotFromEvent'),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Chi2Prob_Min = cms.double(-1.0),
    Pt_Min = cms.double(-1.0)

    checkIsoExtRBarrel            = cms.double(0.4),
    checkIsoInnRBarrel            = cms.double(0.045),
    checkIsoEtaStripBarrel        = cms.double(0.02),
    checkIsoEtRecHitBarrel        = cms.double(0.08),
    checkIsoEtCutBarrel           = cms.double(9.),

    checkIsoExtREndcap            = cms.double(0.4),
    checkIsoInnREndcap            = cms.double(0.07),
    checkIsoEtaStripEndcap        = cms.double(0.02),
    checkIsoEtRecHitEndcap        = cms.double(0.30),
    checkIsoEtCutEndcap           = cms.double(12.)
)
