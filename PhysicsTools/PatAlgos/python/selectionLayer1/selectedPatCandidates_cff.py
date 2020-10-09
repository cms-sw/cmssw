import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.lowPtElectronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.ootPhotonSelector_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import *
#from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cfi import *

# One module to count objects
selectedPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("selectedPatCanddiates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("selectedPatElectrons"),
        cms.InputTag("selectedPatLowPtElectrons"),
        cms.InputTag("selectedPatMuons"),
        cms.InputTag("selectedPatTaus"),
        cms.InputTag("selectedPatPhotons"),
        cms.InputTag("selectedPatOOTPhotons"),
        cms.InputTag("selectedPatJets"),
    )
)

selectedPatCandidatesTask = cms.Task(
    selectedPatElectrons,
    selectedPatLowPtElectrons,
    selectedPatMuons,
    selectedPatTaus,
    selectedPatPhotons,
    selectedPatOOTPhotons,
    selectedPatJets
)

selectedPatCandidates = cms.Sequence(selectedPatCandidateSummary, selectedPatCandidatesTask)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toReplaceWith(selectedPatCandidatesTask, selectedPatCandidatesTask.copyAndExclude([selectedPatOOTPhotons]))
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(selectedPatCandidateSummary.candidates, func = lambda list: list.remove(cms.InputTag("selectedPatOOTPhotons")) )
