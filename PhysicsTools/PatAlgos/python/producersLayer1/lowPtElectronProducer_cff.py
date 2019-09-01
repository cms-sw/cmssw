import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

sourceElectrons = cms.InputTag("lowPtGsfElectrons")

lowPtElectronMatch = electronMatch.clone(
   src = sourceElectrons,
)

patLowPtElectrons = patElectrons.clone(
    # input collections
    electronSource = sourceElectrons,
    genParticleMatch = cms.InputTag("lowPtElectronMatch"),
    # overrides
    addElectronID = cms.bool(False),
    addGenMatch = cms.bool(True),
    addMVAVariables = cms.bool(False),
    addPFClusterIso = cms.bool(False),
    electronIDSources = cms.PSet(),
    computeMiniIso = cms.bool(False),
    isoDeposits = cms.PSet(),
    isolationValues = cms.PSet(),
    isolationValuesNoPFId = cms.PSet(),
    miniIsoParamsB = cms.vdouble(),
    miniIsoParamsE = cms.vdouble(),
    usePfCandidateMultiMap = cms.bool(False),
    # embedding
    embedBasicClusters          = cms.bool(False),
    embedGenMatch               = cms.bool(False),
    embedGsfElectronCore        = cms.bool(False),
    embedGsfTrack               = cms.bool(False),
    embedHighLevelSelection     = cms.bool(False),
    embedPFCandidate            = cms.bool(False),
    embedPflowBasicClusters     = cms.bool(False),
    embedPflowPreshowerClusters = cms.bool(False),
    embedPflowSuperCluster      = cms.bool(False),
    embedPreshowerClusters      = cms.bool(False),
    embedRecHits                = cms.bool(False),
    embedSeedCluster            = cms.bool(False),
    embedSuperCluster           = cms.bool(False),
    embedTrack                  = cms.bool(True),
    )

makePatLowPtElectronsTask = cms.Task(
    lowPtElectronMatch,
    patLowPtElectrons
    )

makePatLowPtElectrons = cms.Sequence(makePatLowPtElectronsTask)


# Modifiers
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(patLowPtElectrons, embedTrack = False)
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17
(run2_miniAOD_80XLegacy | run2_miniAOD_94XFall17).toModify(patLowPtElectrons,
                                                           electronSource = "gedGsfElectrons",
                                                           genParticleMatch = "electronMatch"
                                                           )
