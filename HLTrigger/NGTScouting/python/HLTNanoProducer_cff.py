import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.AK4GenJetFlavourInfos_cfi import *
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import *
from PhysicsTools.JetMCAlgos.TauGenJetsDecayModeSelectorAllHadrons_cfi import *
from PhysicsTools.JetMCAlgos.TauGenJets_cfi import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jetMC_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.taus_cﬀ import *
from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.PatAlgos.slimming.packedGenParticles_cfi import *
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJetsFlavourInfos_cfi import *
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *
from HLTrigger.NGTScouting.hltVertices_cfi import *
from HLTrigger.NGTScouting.hltEGammaPacker_cfi import *
from HLTrigger.NGTScouting.hltPhotons_cfi import *
from HLTrigger.NGTScouting.hltElectrons_cfi import *
from HLTrigger.NGTScouting.hltMuons_cfi import *
from HLTrigger.NGTScouting.hltTracks_cfi import *
from HLTrigger.NGTScouting.hltJets_cfi import *
from HLTrigger.NGTScouting.hltTaus_cfi import *
from HLTrigger.NGTScouting.hltTracksters_cfi import *
from HLTrigger.NGTScouting.hltTICLCandidates_cfi import *
from HLTrigger.NGTScouting.hltTICLSuperClusters_cfi import *
from HLTrigger.NGTScouting.hltLayerClusters_cfi import * 
from HLTrigger.NGTScouting.hltSums_cfi import *
from HLTrigger.NGTScouting.hltTriggerAcceptFilter_cfi import hltTriggerAcceptFilter,dstTriggerAcceptFilter

######################################
# Tables 
######################################

# Produce and store gen particles and gen jets
NanoGenTables = cms.Sequence(
    prunedGenParticlesWithStatusOne
    + prunedGenParticles
    + finalGenParticles
    + genParticleTable
    + genParticlesForJetsNoNu
    + ak4GenJetsNoNu
    + selectedHadronsAndPartonsForGenJetsFlavourInfos
    + packedGenParticles
    + slimmedGenJets
    + ak4GenJetFlavourInfos
    + slimmedGenJetsFlavourInfos
    + genJetTable
    + genJetFlavourTable
    + tauGenJets
    + tauGenJetsForNano
    + tauGenJetsSelectorAllHadrons
    + tauGenJetsSelectorAllHadronsForNano
    + genVisTaus
    + genVisTauTable
)

# Store hlt objects for NGT scouting
NanoHltTables = cms.Sequence(
    hltVertexTable
    + hltPixelVertexTable
    + hltGeneralTrackTable
    + hltGeneralTrackExtTable
    + hltEgammaPacker
    + hltPhotonTable
    + hltElectronTable
    + hltPhase2L3MuonIdTracks
    + hltMuonTable
    + hltPFCandidateTable
    + hltJetTable
    + hltTauTable
    + hltTauExtTable
    + METTable
    + HTTable
)

# Store HGCal lower-level objects
NanoHGCalTables = cms.Sequence(
    hltTrackstersTableSequence
    + hltTiclCandidateTable
    + hltTiclCandidateExtraTable
    + hltTiclSuperClustersTable
)

# Store PixelTracks objects
NanoPixelTables = cms.Sequence(
    hltPixelTrackTable
    + hltPixelTrackExtTable
)

# Store variables and associators for validation purposes
NanoValTables = cms.Sequence(
    hltTiclAssociationsTableSequence
    + hltSimTracksterSequence
    + hltSimTiclCandidateTable
    + hltSimTiclCandidateExtraTable
    + hltLayerClustersTableSequence
)

######################################
# Sequences for Nano flavours
######################################

# NGT Scouting Nano flavour (NANO:@NGTScouting)
dstNanoFlavour = cms.Sequence(
    dstTriggerAcceptFilter
    + NanoHltTables
)

# NGT Scouting Nano flavour with MC/HGCal info (NANO:@NGTScoutingVal)
dstValidationNanoFlavour = cms.Sequence(
    NanoGenTables
    + dstTriggerAcceptFilter
    + NanoHltTables
    + NanoPixelTables
    + NanoHGCalTables
    + NanoValTables
)

# Phase-2 HLT Nano flavour (NANO:@Phase2HLT)
hltNanoFlavour = cms.Sequence(
    NanoHltTables
)

# Phase-2 HLT Nano flavour with MC/HGCal info (NANO:@Phase2HLTVal)
hltValidationNanoFlavour = cms.Sequence(
    NanoGenTables
    + NanoHltTables
    + NanoPixelTables
    + NanoHGCalTables
    + NanoValTables
)

######################################
# Customization
######################################

def hltNanoCustomize(process):

    if hasattr(process, "NANOAODSIMoutput"):
        # process.genJetTable.cut = "pt > 10"
        # process.genJetFlavourTable.deltaR = 0.3
        process.genParticleTable.externalVariables = cms.PSet() # remove iso as external variable from PhysicsTools/NanoAOD/python/genparticles_cff.py:37 (hopefully temporarily)
        process.NANOAODSIMoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )
        process.NANOAODSIMoutput.SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring(
                [p for p in process.paths if p.startswith('HLT_') or p.startswith('MC_') or p.startswith('DST_')]
            )
        )

    return process
