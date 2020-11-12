import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cff import *
from PhysicsTools.PatAlgos.producersLayer1.lowPtElectronProducer_cff import *
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
        cms.InputTag("patLowPtElectrons"),
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
    makePatLowPtElectronsTask,
    makePatMuonsTask,
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

patCandidates = cms.Sequence(patCandidateSummary, patCandidatesTask)

