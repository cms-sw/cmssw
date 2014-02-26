import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs

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



# Collection PFCandidates from a DR=0.8 cone about the jet axis and make new
# faux jets with this collection
from RecoTauTag.RecoTau.RecoTauJetRegionProducer_cfi import \
      RecoTauJetRegionProducer
recoTauAK5PFJets08Region=RecoTauJetRegionProducer.clone(
    src = PFRecoTauPFJetInputs.inputJetCollection
)



# Reconstruct the pi zeros in our pre-selected jets.
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi import \
         ak5PFJetsLegacyHPSPiZeros
ak5PFJetsLegacyHPSPiZeros.jetSrc = PFRecoTauPFJetInputs.inputJetCollection
# import charged hadrons
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cfi import \
          ak5PFJetsRecoTauChargedHadrons

#-------------------------------------------------------------------------------
#------------------ Produce combinatoric base taus------------------------------
#-------------------------------------------------------------------------------
# These jets form the basis of the HPS & TaNC taus.  There are many taus
# produced for each jet, which are cleaned by the respective algorithms.
# We split it into different collections for each different decay mode.

from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import \
        combinatoricRecoTaus

combinatoricRecoTaus.jetSrc = PFRecoTauPFJetInputs.inputJetCollection


#-------------------------------------------------------------------------------
#------------------ HPS Taus ---------------------------------------------------
#-------------------------------------------------------------------------------

from RecoTauTag.Configuration.HPSPFTaus_cff import *

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
ak5PFJetTracksAssociatorAtVertex.jets = PFRecoTauPFJetInputs.inputJetCollection
ak5PFJetTracksAssociatorAtVertex.coneSize = PFRecoTauPFJetInputs.jetConeSize
tautagInfoModifer = cms.PSet(
    name = cms.string("TTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfRecoTauTagInfoProducer"),
)

# Add the modifier to our tau producers
combinatoricRecoTaus.modifiers.append(tautagInfoModifer)

recoTauPileUpVertices = cms.EDFilter(
    "RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)
# import jet filtering sequence
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import tauInputJets

recoTauCommonSequence = cms.Sequence(
    tauInputJets *
    ak5PFJetTracksAssociatorAtVertex *
    recoTauAK5PFJets08Region*
    recoTauPileUpVertices*
    pfRecoTauTagInfoProducer
)



# Produce only classic HPS taus
recoTauClassicHPSSequence = cms.Sequence(
    ak5PFJetsLegacyHPSPiZeros *
    ak5PFJetsRecoTauChargedHadrons *
    combinatoricRecoTaus *
    produceAndDiscriminateHPSPFTaus
)


PFTau = cms.Sequence(
    recoTauCommonSequence *
    recoTauClassicHPSSequence
)

