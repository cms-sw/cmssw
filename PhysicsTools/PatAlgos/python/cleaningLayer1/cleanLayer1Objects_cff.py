import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.cleaningLayer1.electronCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.muonCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.tauCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.photonCleaner_cfi import *
from PhysicsTools.PatAlgos.cleaningLayer1.jetCleaner_cfi import *

from PhysicsTools.PatAlgos.producersLayer1.hemisphereProducer_cfi import *

#FIXME ADD MHT

# One module to count objects
cleanLayer1Summary = cms.EDAnalyzer("CandidateSummaryTable",
    logName = cms.untracked.string("cleanLayer1Objects|PATSummaryTables"),
    candidates = cms.VInputTag(
        cms.InputTag("cleanLayer1Electrons"),
        cms.InputTag("cleanLayer1Muons"),
        cms.InputTag("cleanLayer1Taus"),
        cms.InputTag("cleanLayer1Photons"),
        cms.InputTag("cleanLayer1Jets"),
    )
)


cleanLayer1Objects = cms.Sequence(
    cleanLayer1Muons *        # NOW WE MUST USE '*' AS THE ORDER MATTERS
    cleanLayer1Electrons *
    cleanLayer1Photons *
    cleanLayer1Taus *
    cleanLayer1Jets *
    cleanLayer1Hemispheres * 
    cleanLayer1Summary
)
