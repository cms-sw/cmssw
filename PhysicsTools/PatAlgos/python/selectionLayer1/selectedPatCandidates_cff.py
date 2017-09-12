import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import *
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
        cms.InputTag("selectedPatMuons"),
        cms.InputTag("selectedPatTaus"),
        cms.InputTag("selectedPatPhotons"),
        cms.InputTag("selectedPatOOTPhotons"),
        cms.InputTag("selectedPatJets"),
    )
)

selectedPatCandidatesTask = cms.Task(
    selectedPatElectrons,
    selectedPatMuons,
    selectedPatTaus,
    selectedPatPhotons,
    selectedPatOOTPhotons,
    selectedPatJets
)

selectedPatCandidates = cms.Sequence(selectedPatCandidateSummary, selectedPatCandidatesTask)
