import FWCore.ParameterSet.Config as cms
from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock

tkAssocParamBlock = TrackAssociatorParameterBlock.clone()
tkAssocParamBlock.TrackAssociatorParameters.useMuon = cms.bool(False)
tkAssocParamBlock.TrackAssociatorParameters.useCalo = cms.bool(False)
tkAssocParamBlock.TrackAssociatorParameters.useHO = cms.bool(False)
tkAssocParamBlock.TrackAssociatorParameters.usePreshower = cms.bool(False)
tkAssocParamBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("reducedEcalRecHitsEE")
tkAssocParamBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("reducedEcalRecHitsEB")
tkAssocParamBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("reducedHcalRecHits","hbhereco")
tkAssocParamBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("reducedHcalRecHits","horeco")

isolatedTracks = cms.EDProducer("PATIsolatedTrackProducer",
    tkAssocParamBlock,
    packedPFCandidates = cms.InputTag("packedPFCandidates"),
    lostTracks = cms.InputTag("lostTracks"),
    generalTracks = cms.InputTag("generalTracks"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    caloJets = cms.InputTag("ak4CaloJets"),
    dEdxDataStrip = cms.InputTag("dedxHarmonic2"),
    dEdxDataPixel = cms.InputTag("dedxPixelHarmonic2"),
    dEdxHitInfo = cms.InputTag("dedxHitInfo"),
    usePrecomputedDeDxStrip = cms.bool(True),        # if these are set to True, will get estimated DeDx from DeDxData branches
    usePrecomputedDeDxPixel = cms.bool(True),        # if set to False, will manually compute using dEdxHitInfo
    pT_cut = cms.double(5.0),         # save tracks above this pt
    pT_cut_noIso = cms.double(20.0),  # for tracks with at least this pT, don't apply any iso cut
    pfIsolation_DR = cms.double(0.3),
    pfIsolation_DZ = cms.double(0.1),
    miniIsoParams = cms.vdouble(0.05, 0.2, 10.0), # (minDR, maxDR, kT)
                                                  # dR for miniiso is max(minDR, min(maxDR, kT/pT))
    absIso_cut = cms.double(5.0),
    relIso_cut = cms.double(0.2),
    miniRelIso_cut = cms.double(0.2),

    caloJet_DR = cms.double(0.3),

)
