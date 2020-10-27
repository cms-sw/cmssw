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
recoTauAK4PFJets08Region = RecoTauJetRegionProducer.clone(
    src = PFRecoTauPFJetInputs.inputJetCollection
)

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(recoTauAK4PFJets08Region, minJetPt = 999999.0)

# Reconstruct the pi zeros in our pre-selected jets.
from RecoTauTag.RecoTau.RecoTauPiZeroProducer_cff import ak4PFJetsLegacyHPSPiZeros
ak4PFJetsLegacyHPSPiZeros = ak4PFJetsLegacyHPSPiZeros.clone(
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection
)
# import charged hadrons
from RecoTauTag.RecoTau.PFRecoTauChargedHadronProducer_cff import ak4PFJetsRecoTauChargedHadrons
ak4PFJetsRecoTauChargedHadrons = ak4PFJetsRecoTauChargedHadrons.clone()

#-------------------------------------------------------------------------------
#------------------ Produce combinatoric base taus------------------------------
#-------------------------------------------------------------------------------
# These jets form the basis of the HPS & TaNC taus.  There are many taus
# produced for each jet, which are cleaned by the respective algorithms.
# We split it into different collections for each different decay mode.

from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi import combinatoricRecoTaus, combinatoricModifierConfigs
#-------------------------------------------------------------------------------
#------------------ HPS Taus ---------------------------------------------------
#-------------------------------------------------------------------------------
from RecoTauTag.Configuration.HPSPFTaus_cff import *

combinatoricRecoTaus = combinatoricRecoTaus.clone(
    modifiers = cms.VPSet(combinatoricModifierConfigs),
    jetRegionSrc = "recoTauAK4PFJets08Region",
    jetSrc = PFRecoTauPFJetInputs.inputJetCollection,
    chargedHadronSrc = "ak4PFJetsRecoTauChargedHadrons",
    piZeroSrc = "ak4PFJetsLegacyHPSPiZeros"
)
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify(combinatoricRecoTaus, minJetPt = recoTauAK4PFJets08Region.minJetPt)

#--------------------------------------------------------------------------------
# CV: set mass of tau candidates reconstructed in 1Prong0pi0 decay mode to charged pion mass
combinatoricRecoTaus.modifiers.append(cms.PSet(
    name = cms.string("tau_mass"),
    plugin = cms.string("PFRecoTauMassPlugin"),
    verbosity = cms.int32(0)                                    
))    

#-------------------------------------------------------------------------------
#------------------ PFTauTagInfo workaround ------------------------------------
#-------------------------------------------------------------------------------
# Build the PFTauTagInfos separately, then relink them into the taus.
from RecoTauTag.RecoTau.PFRecoTauTagInfoProducer_cfi import pfRecoTauTagInfoProducer
pfRecoTauTagInfoProducer = pfRecoTauTagInfoProducer.clone(
    PFJetTracksAssociatorProducer = "ak4PFJetTracksAssociatorAtVertex"
)
from RecoJets.JetAssociationProducers.ak4JTA_cff import ak4JetTracksAssociatorAtVertexPF
ak4PFJetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertexPF.clone(
    jets = PFRecoTauPFJetInputs.inputJetCollection
)
tautagInfoModifer = cms.PSet(
    name = cms.string("TTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfRecoTauTagInfoProducer"),
)
combinatoricRecoTaus.modifiers.append(tautagInfoModifer)

recoTauPileUpVertices = cms.EDFilter("RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)

recoTauCommonTask = cms.Task(
    ak4PFJetTracksAssociatorAtVertex,
    recoTauAK4PFJets08Region,
    recoTauPileUpVertices,
    pfRecoTauTagInfoProducer
)
recoTauCommonSequence = cms.Sequence(
    recoTauCommonTask
)

# Produce only classic HPS taus
recoTauClassicHPSTask = cms.Task(
    ak4PFJetsLegacyHPSPiZeros,
    ak4PFJetsRecoTauChargedHadrons,
    combinatoricRecoTaus,
    produceAndDiscriminateHPSPFTausTask
)
recoTauClassicHPSSequence = cms.Sequence(
    recoTauClassicHPSTask
)

PFTauTask = cms.Task(
    recoTauCommonTask,
    recoTauClassicHPSTask
)
PFTau = cms.Sequence(
    PFTauTask
)
