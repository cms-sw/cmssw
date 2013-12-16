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



# Collection PFCandidates from a DR=0.8 cone about the jet axis and make new
# faux jets with this collection
recoTauAK4PFJets08Region = cms.EDProducer(
    "RecoTauJetRegionProducer",
    deltaR = cms.double(0.8),
    src = cms.InputTag("ak4PFJets"),
    pfCandSrc = cms.InputTag("particleFlow"),
    pfCandAssocMapSrc = cms.InputTag("")
)



# Reconstruct the pi zeros in our pre-selected jets.
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi import \
         ak4PFJetsLegacyHPSPiZeros
ak4PFJetsLegacyHPSPiZeros.jetSrc = cms.InputTag("ak4PFJets")
# import charged hadrons
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cfi import \
          ak4PFJetsRecoTauChargedHadrons

#-------------------------------------------------------------------------------
#------------------ Produce combinatoric base taus------------------------------
#-------------------------------------------------------------------------------
# These jets form the basis of the HPS & TaNC taus.  There are many taus
# produced for each jet, which are cleaned by the respective algorithms.
# We split it into different collections for each different decay mode.

from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import \
        combinatoricRecoTaus

combinatoricRecoTaus.jetSrc = cms.InputTag("ak4PFJets")


#-------------------------------------------------------------------------------
#------------------ HPS Taus ---------------------------------------------------
#-------------------------------------------------------------------------------

from RecoTauTag.Configuration.HPSPFTaus_cff import *

combinatoricRecoTaus.piZeroSrc = cms.InputTag("ak4PFJetsLegacyHPSPiZeros")

#-------------------------------------------------------------------------------
#------------------ PFTauTagInfo workaround ------------------------------------
#-------------------------------------------------------------------------------
# Build the PFTauTagInfos separately, then relink them into the taus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import \
        pfRecoTauTagInfoProducer
from RecoJets.JetAssociationProducers.ak4JTA_cff import ak4JetTracksAssociatorAtVertexPF
ak4PFJetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertexPF.clone()
ak4PFJetTracksAssociatorAtVertex.jets = cms.InputTag("ak4PFJets")
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


recoTauCommonSequence = cms.Sequence(
    ak4PFJetTracksAssociatorAtVertex *
    recoTauAK4PFJets08Region*
    recoTauPileUpVertices*
    pfRecoTauTagInfoProducer
)



# Produce only classic HPS taus
recoTauClassicHPSSequence = cms.Sequence(
    ak4PFJetsLegacyHPSPiZeros *
    ak4PFJetsRecoTauChargedHadrons *
    combinatoricRecoTaus *
    produceAndDiscriminateHPSPFTaus
)


PFTau = cms.Sequence(
    recoTauCommonSequence *
    recoTauClassicHPSSequence
)

