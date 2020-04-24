import FWCore.ParameterSet.Config as cms
import copy

#Define the mapping of Decay mode IDs onto the names of trained MVA files Note
#that one category can apply to multiple decay modes, a decay mode can not have
#multiple categories

from RecoTauTag.RecoTau.TauDiscriminatorTools import requireLeadTrack
from RecoTauTag.RecoTau.RecoTauHPSTancTauProdcuer_cfi import combinatoricRecoTausDiscriminationByTaNC

shrinkingConeLeadTrackFinding = copy.deepcopy(requireLeadTrack)
shrinkingConeLeadTrackFinding.leadTrack.Producer = \
        cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackFinding")

shrinkingConePFTauDiscriminationByTaNC = combinatoricRecoTausDiscriminationByTaNC.clone(
    PFTauProducer     = cms.InputTag("shrinkingConePFTauProducer"),
    Prediscriminants  = shrinkingConeLeadTrackFinding,
    discriminantOptions = cms.PSet(),
    dbLabel           = cms.string(""),      # Allow multiple record types
    remapOutput       = cms.bool(True),
    mvas = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("OneProngNoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            mvaLabel = cms.string("OneProngOnePiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            mvaLabel = cms.string("OneProngTwoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("ThreeProngNoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(1),
            mvaLabel = cms.string("ThreeProngOnePiZero"),
        ),
    ),
    prefailValue      = cms.double(-2.0),
)
