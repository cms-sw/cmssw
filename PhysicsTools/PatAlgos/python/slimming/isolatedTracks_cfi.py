import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

TrackAssociatorParameterBlock.TrackAssociatorParameters.useMuon = cms.bool(False)
TrackAssociatorParameterBlock.TrackAssociatorParameters.useCalo = cms.bool(False)
TrackAssociatorParameterBlock.TrackAssociatorParameters.useHO = cms.bool(False)
TrackAssociatorParameterBlock.TrackAssociatorParameters.usePreshower = cms.bool(False)
TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("reducedEcalRecHitsEE")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("reducedEcalRecHitsEB")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("reducedHcalRecHits","hbhereco")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("reducedHcalRecHits","horeco")

isolatedTracks = cms.EDProducer("PATIsolatedTrackProducer",
    TrackAssociatorParameterBlock,
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
    lostTracks = cms.InputTag("lostTracks"),
    generalTracks = cms.InputTag("generalTracks"),
    caloJets = cms.InputTag("ak4CaloJets"),                               
    dEdxInfo = cms.InputTag("dedxHarmonic2"),
    dEdxHitInfo = cms.InputTag("dedxHitInfo"),
    pT_cut = cms.double(5.0),         # save tracks above this pt
    pT_cut_noIso = cms.double(20.0),  # for tracks with at least this pT, don't apply any iso cut
    pfIsolation_DR = cms.double(0.3),
    pfIsolation_DZ = cms.double(0.1),
    miniIsoParams = cms.vdouble(0.05, 0.2, 10.0), # (minDR, maxDR, kT)
                                                  # dR for miniiso is max(minDR, min(maxDR, kT/pT))
    absIso_cut = cms.double(5.0),
    relIso_cut = cms.double(0.2),
    miniRelIso_cut = cms.double(0.2),

    # absIso_cut = cms.double(999999.0),
    # relIso_cut = cms.double(999999.0),
    # miniRelIso_cut = cms.double(999999.0),

    caloJet_DR = cms.double(0.3),

)
