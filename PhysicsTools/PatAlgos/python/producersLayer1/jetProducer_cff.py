import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.recoLayer0.bTagging_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff import *
from PhysicsTools.PatAlgos.recoLayer0.jetCorrections_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *

## for scheduled mode
makePatJetsTask = cms.Task(
    patJetCorrectionsTask,
    patJetCharge,
    patJetPartonMatch,
    patJetGenJetMatch,
    patJetFlavourIdLegacyTask,
    patJetFlavourIdTask,
    patJets
    )

from PhysicsTools.PatAlgos.producersHeavyIons.heavyIonJets_cff import *
_makePatJetsTaskHI = cms.Task(
    recoJetsHIpostAODTask,
    makePatJetsTask.copy()
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(makePatJetsTask, _makePatJetsTaskHI)

makePatJets = cms.Sequence(makePatJetsTask)

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import * #pfImpactParameterTagInfos
from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import * #pfSecondaryVertexTagInfos
from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import * #pfInclusiveSecondaryVertexFinderTagInfos
from RecoBTag.Combined.deepFlavour_cff import * #pfDeepCSVTask

#make a copy to avoid labels and substitution problems
_makePatJetsWithDeepFlavorTask = makePatJetsTask.copy()
_makePatJetsWithDeepFlavorTask.add(
    pfImpactParameterTagInfos, 
    pfSecondaryVertexTagInfos,
    pfInclusiveSecondaryVertexFinderTagInfos,
    pfDeepCSVTask
)

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toReplaceWith(
    makePatJetsTask, _makePatJetsWithDeepFlavorTask
)


