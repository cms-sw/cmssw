import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
# to be able to run PFTau sequence standalone on AOD
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import TransientTrackBuilderESProducer

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
from RecoTauTag.RecoTau.RecoTauJetRegionProducer_cfi import RecoTauJetRegionProducer
recoTauAK4PFJets08Region76xReMiniAOD = RecoTauJetRegionProducer.clone(
    src = PFRecoTauPFJetInputs.inputJetCollection
)

# Reconstruct the pi zeros in our pre-selected jets.
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi import ak4PFJetsLegacyHPSPiZeros
ak4PFJetsLegacyHPSPiZeros76xReMiniAOD = ak4PFJetsLegacyHPSPiZeros.clone()
ak4PFJetsLegacyHPSPiZeros76xReMiniAOD.jetSrc = PFRecoTauPFJetInputs.inputJetCollection

# import charged hadrons
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cfi import ak4PFJetsRecoTauChargedHadrons
ak4PFJetsRecoTauChargedHadrons76xReMiniAOD = ak4PFJetsRecoTauChargedHadrons.clone()

#-------------------------------------------------------------------------------
#------------------ Produce combinatoric base taus------------------------------
#-------------------------------------------------------------------------------
# These jets form the basis of the HPS & TaNC taus.  There are many taus
# produced for each jet, which are cleaned by the respective algorithms.
# We split it into different collections for each different decay mode.

from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import combinatoricRecoTaus
combinatoricRecoTaus76xReMiniAOD = combinatoricRecoTaus.clone()
combinatoricRecoTaus76xReMiniAOD.jetRegionSrc = cms.InputTag("recoTauAK4PFJets08Region76xReMiniAOD")
combinatoricRecoTaus76xReMiniAOD.jetSrc = PFRecoTauPFJetInputs.inputJetCollection

#--------------------------------------------------------------------------------
# CV: disable reconstruction of 3Prong1Pi0 tau candidates
combinatoricRecoTaus76xReMiniAOD.builders[0].decayModes.remove(combinatoricRecoTaus76xReMiniAOD.builders[0].decayModes[6])
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# CV: set mass of tau candidates reconstructed in 1Prong0pi0 decay mode to charged pion mass
combinatoricRecoTaus76xReMiniAOD.modifiers.append(cms.PSet(
    name = cms.string("tau_mass"),
    plugin = cms.string("PFRecoTauMassPlugin"),
    verbosity = cms.int32(0)                                    
))    
#--------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#------------------ HPS Taus ---------------------------------------------------
#-------------------------------------------------------------------------------

from RecoTauTag.Configuration.HPSPFTaus_reminiAOD_cff import *

combinatoricRecoTaus76xReMiniAOD.chargedHadronSrc = cms.InputTag("ak4PFJetsRecoTauChargedHadrons76xReMiniAOD")
combinatoricRecoTaus76xReMiniAOD.piZeroSrc = cms.InputTag("ak4PFJetsLegacyHPSPiZeros76xReMiniAOD")

#-------------------------------------------------------------------------------
#------------------ PFTauTagInfo workaround ------------------------------------
#-------------------------------------------------------------------------------
# Build the PFTauTagInfos separately, then relink them into the taus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import pfRecoTauTagInfoProducer
pfRecoTauTagInfoProducer76xReMiniAOD = pfRecoTauTagInfoProducer.clone()
pfRecoTauTagInfoProducer76xReMiniAOD.PFJetTracksAssociatorProducer = cms.InputTag("ak4PFJetTracksAssociatorAtVertex76xReMiniAOD")

from RecoJets.JetAssociationProducers.ak4JTA_cff import ak4JetTracksAssociatorAtVertexPF
ak4PFJetTracksAssociatorAtVertex76xReMiniAOD = ak4JetTracksAssociatorAtVertexPF.clone()
ak4PFJetTracksAssociatorAtVertex76xReMiniAOD.jets = PFRecoTauPFJetInputs.inputJetCollection
tautagInfoModifer76xReMiniAOD = cms.PSet(
    name = cms.string("TTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfRecoTauTagInfoProducer76xReMiniAOD"),
)

# Add the modifier to our tau producers
hasTTIworkaround = False
for modifier in combinatoricRecoTaus76xReMiniAOD.modifiers:
    if hasattr(modifier, "name") and modifier.name.value() == "TTIworkaround":
        hasTTIworkaround = True
if not hasTTIworkaround:
    combinatoricRecoTaus76xReMiniAOD.modifiers.append(tautagInfoModifer76xReMiniAOD)
##combinatoricRecoTaus76xReMiniAOD.modifiers.append(tautagInfoModifer76xReMiniAOD)

recoTauPileUpVertices76xReMiniAOD = cms.EDFilter("RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)

recoTauCommonSequence76xReMiniAOD = cms.Sequence(
    ak4PFJetTracksAssociatorAtVertex76xReMiniAOD *
    recoTauAK4PFJets08Region76xReMiniAOD *
    recoTauPileUpVertices76xReMiniAOD *
    pfRecoTauTagInfoProducer76xReMiniAOD
)

# Produce only classic HPS taus
recoTauClassicHPSSequence76xReMiniAOD = cms.Sequence(
    ak4PFJetsLegacyHPSPiZeros76xReMiniAOD *
    ak4PFJetsRecoTauChargedHadrons76xReMiniAOD *
    combinatoricRecoTaus76xReMiniAOD *
    produceAndDiscriminateHPSPFTaus76xReMiniAOD
)

PFTau76xReMiniAOD = cms.Sequence(
    recoTauCommonSequence76xReMiniAOD *
    recoTauClassicHPSSequence76xReMiniAOD
)

