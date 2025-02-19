import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.cleaningLayer1.electronCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.muonCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.tauCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.photonCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.jetCleaner_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cfi import *
#FIXME ADD MHT

# One module to count objects
cleanPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("cleanPatCandidates|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("cleanPatElectrons"),
        cms.InputTag("cleanPatMuons"),
        cms.InputTag("cleanPatTaus"),
        cms.InputTag("cleanPatPhotons"),
        cms.InputTag("cleanPatJets"),
    )
)


cleanPatCandidates = cms.Sequence(
    cleanPatMuons     *        # NOW WE MUST USE '*' AS THE ORDER MATTERS
    cleanPatElectrons *
    cleanPatPhotons   *
    cleanPatTaus      *
    cleanPatJets      *
    cleanPatCandidateSummary
)
