import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from HLTrigger.NGTScouting.hltVertices_cfi import *
from HLTrigger.NGTScouting.hltEGammaPacker_cfi import *
from HLTrigger.NGTScouting.hltPhotons_cfi import *
from HLTrigger.NGTScouting.hltElectrons_cfi import *
from HLTrigger.NGTScouting.hltMuons_cfi import *
from HLTrigger.NGTScouting.hltTracks_cfi import *
from HLTrigger.NGTScouting.hltJets_cfi import *
from HLTrigger.NGTScouting.hltTriggerAcceptFilter_cfi import hltTriggerAcceptFilter,dstTriggerAcceptFilter

hltNanoProducer = cms.Sequence(
    prunedGenParticles
    + finalGenParticles
    + genParticleTable
    + hltTriggerAcceptFilter
    + hltVertexTable
    + hltPixelTrackTable
    + hltGeneralTrackTable
    + hltEgammaPacker
    + hltPhotonTable
    + hltElectronTable
    + hltPhase2L3MuonIdTracks
    + hltMuonTable
    + hltPFCandidateTable
    + hltJetTable
)

dstNanoProducer = cms.Sequence(
    prunedGenParticles
    + finalGenParticles
    + genParticleTable
    + dstTriggerAcceptFilter
    + hltVertexTable
    + hltPixelTrackTable
    + hltGeneralTrackTable
    + hltEgammaPacker
    + hltPhotonTable
    + hltElectronTable
    + hltPhase2L3MuonIdTracks
    + hltMuonTable
    + hltPFCandidateTable
    + hltJetTable
)

def hltNanoCustomize(process):

    if hasattr(process, "NANOAODSIMoutput"):
        process.prunedGenParticles.src = "genParticles"
        process.genParticleTable.externalVariables = cms.PSet() # remove iso as external variable from PhysicsTools/NanoAOD/python/genparticles_cff.py:37 (hopefully temporarily)
        process.NANOAODSIMoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )
        process.NANOAODSIMoutput.SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring(
                [p for p in process.paths if p.startswith('HLT_') or p.startswith('DST_')]
            )
        )

    return process
