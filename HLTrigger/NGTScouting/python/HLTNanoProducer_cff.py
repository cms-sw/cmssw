import FWCore.ParameterSet.Config as cms

from PhysicsTools.JetMCAlgos.AK4GenJetFlavourInfos_cfi import *
from PhysicsTools.JetMCAlgos.HadronAndPartonSelector_cfi import *
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.jetMC_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
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

NanoGenTable = cms.Sequence(
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
)

hltNanoProducer = cms.Sequence(
    NanoGenTable
    #+ hltTriggerAcceptFilter
    + hltVertexTable
    + hltPixelTrackTable
    + hltPixelVertexTable
    + hltGeneralTrackTable
    + hltEgammaPacker
    + hltPhotonTable
    + hltElectronTable
    + hltPhase2L3MuonIdTracks
    + hltMuonTable
    + hltPFCandidateTable
    + hltJetTable
    + hltTrackstersTableSequence
    + hltTiclCandidateTable
    + hltTiclCandidateExtraTable
    + hltTiclSuperClustersTable
    + hltTauTable
    + hltTauExtTable
    + METTable
    + HTTable
)

dstNanoProducer = cms.Sequence(
    NanoGenTable
    + dstTriggerAcceptFilter
    + hltVertexTable
    + hltPixelTrackTable
    + hltPixelVertexTable
    + hltGeneralTrackTable
    + hltEgammaPacker
    + hltPhotonTable
    + hltElectronTable
    + hltPhase2L3MuonIdTracks
    + hltMuonTable
    + hltPFCandidateTable
    + hltJetTable
    + hltTauTable
    + hltTrackstersTableSequence
    + hltTiclCandidateTable
    + hltTiclCandidateExtraTable
    + hltTiclSuperClustersTable
    + hltTauExtTable
    + METTable
    + HTTable
)

trackingExtraNanoProducer = cms.Sequence(
    hltPixelTrackExtTable+
    hltGeneralTrackExtTable
)

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

def hltNanoValCustomize(process):
    if hasattr(process, "dstNanoProducer"):

        process.dstNanoProducer += (process.hltTiclAssociationsTableSequence +
                                    process.hltSimTracksterSequence +
                                    process.hltSimTiclCandidateTable +
                                    process.hltSimTiclCandidateExtraTable +
                                    process.hltLayerClustersTableSequence  +
                                    process.trackingExtraNanoProducer)

    return process
