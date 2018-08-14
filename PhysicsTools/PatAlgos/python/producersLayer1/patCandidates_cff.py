import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *
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
        cms.InputTag("patMuons"),
        cms.InputTag("patTaus"),
        cms.InputTag("patPhotons"),
        cms.InputTag("patOOTPhotons"),
        cms.InputTag("patJets"),
        cms.InputTag("patMETs"),
    )
)

patCandidatesTask = cms.Task(
    makePatElectronsTask,
    makePatMuonsTask,
    makePatTausTask,
    makePatPhotonsTask,
    makePatOOTPhotonsTask,
    makePatJetsTask,
    makePatMETsTask
)
patCandidates = cms.Sequence(patCandidateSummary, patCandidatesTask)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(patCandidatesTask, patCandidatesTask.copyAndExclude([makePatOOTPhotonsTask]))
pp_on_AA_2018.toModify(patCandidateSummary.candidates, func = lambda list: list.remove(cms.InputTag("patOOTPhotons")) )
