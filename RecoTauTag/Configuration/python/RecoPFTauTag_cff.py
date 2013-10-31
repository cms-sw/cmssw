import FWCore.ParameterSet.Config as cms

#-------------------------------------------------------------------------------
#------------------ Jet Production and Preselection-----------------------------
#-------------------------------------------------------------------------------
# Apply a base selection to the jets.  The acceptance selection takes only jets
# with pt > 5 and abs(eta) < 2.5.  The preselection selects jets that have at
# least one constituent with pt > 5.  This cut should be 100% efficient w.r.t a
# lead pion selection.
#
# After the basic preselection has been applied to the jets, the pizeros inside
# the jet are reconstructed.
#-------------------------------------------------------------------------------

# Produce the jets that form the base of PFTaus
#from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets

# Reconstruct the pi zeros in our pre-selected jets.
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi import \
     ak5PFJetsRecoTauPiZeros
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cfi import \
     ak5PFJetsRecoTauChargedHadrons

# Collection PFCandidates from a DR=0.8 cone about the jet axis and make new
# faux jets with this collection
recoTauAK5PFJets08Region = cms.EDProducer(
    "RecoTauJetRegionProducer",
    deltaR = cms.double(0.8),
    src = cms.InputTag("ak5PFJets"),
    pfCandSrc = cms.InputTag("particleFlow"),
    pfCandAssocMapSrc = cms.InputTag("")
)

# The computation of the lead track signed transverse impact parameter depends
# on the transient tracks
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import \
        TransientTrackBuilderESProducer

# Only reconstruct the preselected jets
ak5PFJetsRecoTauPiZeros.jetSrc = cms.InputTag("ak5PFJets")

#-------------------------------------------------------------------------------
#------------------ Fixed Cone Taus --------------------------------------------
#-------------------------------------------------------------------------------
#from RecoTauTag.Configuration.FixedConePFTaus_cff import *

#-------------------------------------------------------------------------------
#------------------ Shrinking Cone Taus ----------------------------------------
#-------------------------------------------------------------------------------
#from RecoTauTag.Configuration.ShrinkingConePFTaus_cff import *
# Use the legacy PiZero reconstruction for shrinking cone taus
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi import \
        ak5PFJetsLegacyTaNCPiZeros, ak5PFJetsLegacyHPSPiZeros

#ak5PFJetsLegacyTaNCPiZeros.jetSrc = cms.InputTag("ak5PFJets")

#shrinkingConePFTauProducer.piZeroSrc = cms.InputTag(
#    "ak5PFJetsLegacyTaNCPiZeros")

#-------------------------------------------------------------------------------
#------------------ Produce combinatoric base taus------------------------------
#-------------------------------------------------------------------------------
# These jets form the basis of the HPS & TaNC taus.  There are many taus
# produced for each jet, which are cleaned by the respective algorithms.
# We split it into different collections for each different decay mode.

from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import \
        combinatoricRecoTaus

combinatoricRecoTaus.jetSrc = cms.InputTag("ak5PFJets")
combinatoricRecoTaus.piZeroSrc = cms.InputTag("ak5PFJetsRecoTauPiZeros")

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import \
        pfRecoTauDiscriminationByLeadingPionPtCut
# Common discrimination by lead pion
combinatoricRecoTausDiscriminationByLeadingPionPtCut = \
        pfRecoTauDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("combinatoricRecoTaus")
        )

#-------------------------------------------------------------------------------
#------------------ HPS Taus ---------------------------------------------------
#-------------------------------------------------------------------------------

from RecoTauTag.Configuration.HPSPFTaus_cff import *
from RecoTauTag.Configuration.HPSTancTaus_cff import *
ak5PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag("ak5PFJets")

# FIXME remove this once final pi zero reco is decided
combinatoricRecoTaus.piZeroSrc = cms.InputTag("ak5PFJetsLegacyHPSPiZeros")

#-------------------------------------------------------------------------------
#------------------ PFTauTagInfo workaround ------------------------------------
#-------------------------------------------------------------------------------
# Build the PFTauTagInfos separately, then relink them into the taus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import \
        pfRecoTauTagInfoProducer
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi \
        import ic5PFJetTracksAssociatorAtVertex
ak5PFJetTracksAssociatorAtVertex = ic5PFJetTracksAssociatorAtVertex.clone()
ak5PFJetTracksAssociatorAtVertex.jets = cms.InputTag("ak5PFJets")
tautagInfoModifer = cms.PSet(
    name = cms.string("TTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfRecoTauTagInfoProducer"),
)

# Add the modifier to our tau producers
#shrinkingConePFTauProducerSansRefs.modifiers.append(tautagInfoModifer)
combinatoricRecoTaus.modifiers.append(tautagInfoModifer)

recoTauPileUpVertices = cms.EDFilter(
    "RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)


recoTauCommonSequence = cms.Sequence(
    ak5PFJetTracksAssociatorAtVertex *
    recoTauAK5PFJets08Region*
    recoTauPileUpVertices*
    pfRecoTauTagInfoProducer
)


# Not run in RECO, but included for the benefit of PAT
# recoTauClassicFixedConeSequence = cms.Sequence(
#     recoTauCommonSequence *
#     ak5PFJetsRecoTauPiZeros *
#     produceAndDiscriminateFixedConePFTaus
# )

# Produce only classic HPS taus
recoTauClassicHPSSequence = cms.Sequence(
    recoTauCommonSequence *
    ak5PFJetsLegacyHPSPiZeros *
    ak5PFJetsRecoTauChargedHadrons *
    combinatoricRecoTaus *
    produceAndDiscriminateHPSPFTaus
)

# Produce only classic shrinking cone taus (+ TaNC)
# recoTauClassicShrinkingConeSequence = cms.Sequence(
#     recoTauCommonSequence *
#     ak5PFJetsRecoTauPiZeros *
#     produceAndDiscriminateShrinkingConePFTaus
# )

# recoTauClassicShrinkingConeMVASequence = cms.Sequence(
#     produceShrinkingConeDiscriminationByTauNeuralClassifier
# )

# Produce hybrid algorithm taus
# recoTauHPSTancSequence = cms.Sequence(
#     recoTauCommonSequence *
#     ak5PFJetsLegacyHPSPiZeros *
#     combinatoricRecoTaus *
#     hpsTancTauInitialSequence *
#     hpsTancTauDiscriminantSequence
# )

PFTau = cms.Sequence(
    # Jet production
    recoTauCommonSequence *
    # Make shrinking cone taus
#    recoTauClassicShrinkingConeSequence *
    # Make classic HPS taus
    recoTauClassicHPSSequence
)

# Check if we want to run the MVA dependent stuff.  This is disabled in some
# versions of 3_11_1 due to a TMVA issue.
#from RecoTauTag.Configuration.RecoTauMVAConfiguration_cfi \
#        import recoTauEnableMVA
#
#if recoTauEnableMVA:
#    # Enable shrinking cone tanc discriminators
#    PFTau += recoTauClassicShrinkingConeMVASequence
#    # Make hybrid algo taus
#    PFTau += recoTauHPSTancSequence
