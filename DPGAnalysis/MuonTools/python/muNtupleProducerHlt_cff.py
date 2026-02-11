import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from DPGAnalysis.MuonTools.nano_mu_hlt_cff import *


hltMuNanoProducer = cms.Sequence(
    prunedGenParticles
    + finalGenParticles
    + genParticleTable
    + hltMuonTriggerProducers
)

hltMuNanoProducerForTraining = cms.Sequence(
    prunedGenParticles
    + finalGenParticles
    + genParticleTable
)

_hltMuNanoProducerTrainingPixel = hltMuNanoProducerForTraining.copy()
_hltMuNanoProducerTrainingPixel += hltMuonTriggerProducersForTrainingPixel
from Configuration.ProcessModifiers.phase2MuonPixelTracksSelector_cff import phase2MuonPixelTracksSelector
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2MuonPixelTracksSelector | ngtScouting).toReplaceWith(
    hltMuNanoProducerForTraining,
    _hltMuNanoProducerTrainingPixel
)

_hltMuNanoProducerTrainingSeeds = hltMuNanoProducerForTraining.copy()
_hltMuNanoProducerTrainingSeeds += hltMuonTriggerProducersForTrainingSeeds
from Configuration.ProcessModifiers.phase2MuonSeedsSelector_cff import phase2MuonSeedsSelector
(phase2MuonSeedsSelector).toReplaceWith(
    hltMuNanoProducerForTraining,
    _hltMuNanoProducerTrainingSeeds
)

def hltMuNanoCustomize(process):

    if hasattr(process, "NANOAODSIMoutput"):
        process.prunedGenParticles.src = "genParticles"
        process.genParticleTable.externalVariables = cms.PSet() # remove iso as external variable from PhysicsTools/NanoAOD/python/genparticles_cff.py:37 (hopefully temporarily)
        process.NANOAODSIMoutput.outputCommands.append(
            "keep nanoaodFlatTable_*Table*_*_*"
        )
        process.NANOAODSIMoutput.outputCommands.append("drop edmTriggerResults_*_*_*")

    return process
