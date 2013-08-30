import FWCore.ParameterSet.Config as cms

'''

Configuration for ChargedHadron producer plugins.

Author: Christian Veelken, LLR


'''

from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts

# Produce a ChargedHadron candidate for each charged PFCandidate
chargedPFCandidates = cms.PSet(
    name = cms.string("chargedPFCandidates"),
    plugin = cms.string("PFRecoTauChargedHadronFromPFCandidatePlugin"),
    # process PFChargedHadrons and PFElectrons
    # (numbering scheme defined in DataFormats/ParticleFlowCandidate/interface/PFCandidate.h)
    chargedHadronCandidatesParticleIds = cms.vint32(1, 2, 3), # PFCandidate types = h, e, mu 
    qualityCuts = PFTauQualityCuts,
    dRmergeNeutralHadronWrtChargedHadron = cms.double(0.005),
    dRmergeNeutralHadronWrtNeutralHadron = cms.double(0.010),
    dRmergeNeutralHadronWrtElectron = cms.double(0.05),
    dRmergeNeutralHadronWrtOther = cms.double(0.005),
    minBlockElementMatchesNeutralHadron = cms.int32(2),
    maxUnmatchedBlockElementsNeutralHadron = cms.int32(1),
    dRmergePhotonWrtChargedHadron = cms.double(0.005),
    dRmergePhotonWrtNeutralHadron = cms.double(0.010),
    dRmergePhotonWrtElectron = cms.double(0.005),
    dRmergePhotonWrtOther = cms.double(0.005),    
    minBlockElementMatchesPhoton = cms.int32(2),
    maxUnmatchedBlockElementsPhoton = cms.int32(1),
    minMergeNeutralHadronEt = cms.double(0.),
    minMergeGammaEt = cms.double(0.),
    minMergeChargedHadronPt = cms.double(100.)
)

# Produce a ChargedHadron candidate for each reco::Track
# (overlap with charged PFCandidate is removed by PFRecoTauChargedHadronProducer module)
tracks = cms.PSet(
    name = cms.string("tracks"),
    plugin = cms.string("PFRecoTauChargedHadronFromTrackPlugin"),
    srcTracks = cms.InputTag('generalTracks'),
    dRcone = cms.double(0.5),
    qualityCuts = PFTauQualityCuts,
    dRmergeNeutralHadron = cms.double(0.10),
    dRmergePhoton = cms.double(0.05),
    minMergeNeutralHadronEt = cms.double(0.),
    minMergeGammaEt = cms.double(0.),
    minMergeChargedHadronPt = cms.double(100.)
)

# Produce a ChargedHadron candidate for high Pt PFNeutralHadrons
PFNeutralHadrons = chargedPFCandidates.clone(
    name = cms.string("PFNeutralHadrons"),
    plugin = cms.string("PFRecoTauChargedHadronFromPFCandidatePlugin"),
    # process PFNeutralHadrons
    # (numbering scheme defined in DataFormats/ParticleFlowCandidate/interface/PFCandidate.h)
    chargedHadronCandidatesParticleIds = cms.vint32(5),
    minMergeChargedHadronPt = cms.double(0.)
)
