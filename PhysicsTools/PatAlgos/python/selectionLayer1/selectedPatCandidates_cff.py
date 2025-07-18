import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.lowPtElectronSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.displacedMuonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.photonSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.ootPhotonSelector_cff import *
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi import selectedPatJets as _selectedPatJets
#from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cfi import *

selectedPatJets = _selectedPatJets.clone()

# One module to count objects
selectedPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("selectedPatCanddiates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("selectedPatElectrons"),
        cms.InputTag("selectedPatLowPtElectrons"),
        cms.InputTag("selectedPatMuons"),
        cms.InputTag("selectedPatDisplacedMuons"),
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
    selectedPatDisplacedMuons,
    selectedPatTaus,
    selectedPatPhotons,
    selectedPatOOTPhotons,
    selectedPatJets
)

selectedPatCandidates = cms.Sequence(selectedPatCandidateSummary, selectedPatCandidatesTask)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(selectedPatCandidatesTask, selectedPatCandidatesTask.copyAndExclude([selectedPatOOTPhotons]))
pp_on_AA.toModify(selectedPatCandidateSummary.candidates, func = lambda list: list.remove(cms.InputTag("selectedPatOOTPhotons")) )

(pp_on_AA).toReplaceWith(
    selectedPatCandidatesTask,
    selectedPatCandidatesTask.copyAndExclude([selectedPatLowPtElectrons])).toModify(
        selectedPatCandidateSummary.candidates,
        func = lambda list: list.remove(cms.InputTag("selectedPatLowPtElectrons")) )

from Configuration.ProcessModifiers.run2_miniAOD_UL_cff import run2_miniAOD_UL
from Configuration.Eras.Era_Run2_2016_HIPM_cff import Run2_2016_HIPM
(pp_on_AA | run2_miniAOD_UL | Run2_2016_HIPM).toReplaceWith(selectedPatCandidatesTask,
                                                   selectedPatCandidatesTask.copyAndExclude([selectedPatDisplacedMuons]))
(pp_on_AA | run2_miniAOD_UL | Run2_2016_HIPM).toModify(selectedPatCandidateSummary.candidates,
                                              func = lambda list: list.remove(cms.InputTag("selectedPatDisplacedMuons")) )

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(selectedPatCandidatesTask, selectedPatCandidatesTask.copyAndExclude([selectedPatDisplacedMuons]))
