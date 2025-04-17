import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from HLTrigger.NGTScouting.hltVertices_cfi import *
from HLTrigger.NGTScouting.hltEGammaPacker_cfi import *
from HLTrigger.NGTScouting.hltPhotons_cfi import *
from HLTrigger.NGTScouting.hltElectrons_cfi import *

hltNanoProducer = cms.Sequence(
    hltVertexTable +
    hltEgammaPacker +
    hltPhotonTable +
    hltElectronTable
)

def hltNanoCustomize(process):

    if hasattr(process, "NANOAODSIMoutput"):
        process.prunedGenParticles.src = "genParticles"
        process.genParticleTable.externalVariables = cms.PSet() # remove iso as external variable from PhysicsTools/NanoAOD/python/genparticles_cff.py:37 (hopefully temporarily)
        process.NANOAODSIMoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )

    return process