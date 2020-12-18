import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import *

sourceElectrons = cms.InputTag("lowPtGsfElectrons")

lowPtElectronMatch = electronMatch.clone(
   src = sourceElectrons,
)

patLowPtElectrons = patElectrons.clone(

    # Input collection
    electronSource = sourceElectrons,

    # MC matching
    genParticleMatch = "lowPtElectronMatch",

    # Electron ID
    addElectronID = True,
    electronIDSources = dict(
        unbiased = cms.InputTag("rekeyLowPtGsfElectronSeedValueMaps:unbiased"),
        ptbiased = cms.InputTag("rekeyLowPtGsfElectronSeedValueMaps:ptbiased"),
        ID       = cms.InputTag("lowPtGsfElectronID"),
    ),

    # Embedding of RECO/AOD items

    # Embedding of RECO/AOD items
    embedTrack                  = True,
    embedGsfElectronCore        = True,
    embedGsfTrack               = True,
    embedSuperCluster           = True,
    embedSeedCluster            = True,
    embedBasicClusters          = True,
    embedPreshowerClusters      = False,
    embedRecHits                = False,
    embedPflowSuperCluster      = False,
    embedPflowBasicClusters     = False,
    embedPflowPreshowerClusters = False,
    embedPFCandidate            = False,

    # Miscellaneous flags
    addMVAVariables         = False,
    embedHighLevelSelection = False,
    isoDeposits             = cms.PSet(),
    isolationValues         = cms.PSet(),
    isolationValuesNoPFId   = cms.PSet(),

)

makePatLowPtElectronsTask = cms.Task(
    lowPtElectronMatch,
    patLowPtElectrons,
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

# Schedule rekeying of seed BDT ValueMaps by reco::GsfElectron for run2_miniAOD_UL
from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronSeedValueMaps_cff import rekeyLowPtGsfElectronSeedValueMaps
from RecoEgamma.EgammaElectronProducers.lowPtGsfElectronID_cfi import lowPtGsfElectronID
_makePatLowPtElectronsTask = makePatLowPtElectronsTask.copy()
_makePatLowPtElectronsTask.add(rekeyLowPtGsfElectronSeedValueMaps)
_makePatLowPtElectronsTask.add(lowPtGsfElectronID)
run2_miniAOD_UL.toReplaceWith(makePatLowPtElectronsTask,_makePatLowPtElectronsTask)
