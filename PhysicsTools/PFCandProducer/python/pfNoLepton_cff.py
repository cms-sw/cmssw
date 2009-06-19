import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cff  import *

from PhysicsTools.PFCandProducer.pfTopProjectionIsolatedPFCandidatesOnPileUpPFCandidates_cfi import pfTopProjectionIsolatedPFCandidatesOnPFCandidates as tp
muonsOnNoPileUp = tp.clone()
muonsOnNoPileUp.name = 'muonsOnNoPileUp'
muonsOnNoPileUp.topCollection = 'pfMuons'
muonsOnNoPileUp.bottomCollection = 'pileUpOnPFCandidates'

# not used yet
electronsOnNoMuons = muonsOnNoPileUp.clone()
electronsOnNoMuons.name = 'electronsOnNoMuons'
electronsOnNoMuons.topCollection = 'pfElectrons'
electronsOnNoMuons.bottomCollection = 'muonsOnNoPileUp'

dump = cms.EDAnalyzer("EventContentAnalyzer")

pfNoLeptonSequence = cms.Sequence(
    muonsOnNoPileUp
    )

