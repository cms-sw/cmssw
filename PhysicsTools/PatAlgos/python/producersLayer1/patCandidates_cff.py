import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.lowPtElectronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.displacedMuonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.ootPhotonProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cff import *

## module to count objects
patCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("patCandidates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("patElectrons"),
        cms.InputTag("patLowPtElectrons"),
        cms.InputTag("patMuons"),
        cms.InputTag("patDisplacedMuons"),
        cms.InputTag("patTaus"),
        cms.InputTag("patPhotons"),
        cms.InputTag("patOOTPhotons"),
        cms.InputTag("patJets"),
        cms.InputTag("patMETs"),
    )
)

patCandidatesTask = cms.Task(
    makePatElectronsTask,
    makePatLowPtElectronsTask,
    makePatMuonsTask,
    makePatDisplacedMuonsTask,
    makePatTausTask,
    makePatPhotonsTask,
    makePatOOTPhotonsTask,
    makePatJetsTask,
    makePatMETsTask
)

_patCandidatesTask = patCandidatesTask.copy()
_patCandidatesTask.remove(makePatOOTPhotonsTask)
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(patCandidatesTask, _patCandidatesTask)
pp_on_AA.toModify(patCandidateSummary.candidates, func = lambda list: list.remove(cms.InputTag("patOOTPhotons")) )

(pp_on_AA).toReplaceWith(
    patCandidatesTask,
    patCandidatesTask.copyAndExclude([makePatLowPtElectronsTask])).toModify(
        patCandidateSummary.candidates,
        func = lambda list: list.remove(cms.InputTag("patLowPtElectrons")) )

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM
(pp_on_AA | run2_miniAOD_UL | Run2_2016_HIPM).toReplaceWith(
    patCandidatesTask,
    patCandidatesTask.copyAndExclude([makePatDisplacedMuonsTask])).toModify(
        patCandidateSummary.candidates,
        func = lambda list: list.remove(cms.InputTag("patDisplacedMuons")) )

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(patCandidatesTask, patCandidatesTask.copyAndExclude([makePatDisplacedMuonsTask]))

patCandidates = cms.Sequence(patCandidateSummary, patCandidatesTask)

