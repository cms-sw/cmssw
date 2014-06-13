import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadPion
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import pfRecoTauDiscriminationByIsolation

# Cut on sum pt < 8GeV  isolation tracks.

pfRecoTauDiscriminationByIsolationChargedSumPt = pfRecoTauDisciminationByIsolation.clone(
    PFTauProducer = cms.InputTag('pfRecoTauProducer'),

    # Require leading pion ensures that: theee is at least one track above
    # threshold (0.5 GeV) in the signal cone a track in the signal cone has
    # pT > 5 GeV
    Prediscriminants = requireLeadPion,

    # Select which collections to use for isolation.
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),

    applyOccupancyCut = cms.bool(False),
    maximumOccupancy = cms.uint32(1),

    applySumPtCut = cms.bool(True),
    maximumSumPtCut = cms.double(8.0),

    applyRelativeSumPtCut = cms.bool(False),
    relativeSumPtCut = cms.double(0.0),

    # Set the standard quality cuts on the isolation candidates
    qualityCuts = PFTauQualityCuts,
    PVProducer = PFTauQualityCuts.primaryVertexSrc  # need for Q cuts
)
