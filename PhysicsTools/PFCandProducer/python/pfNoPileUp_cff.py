import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfTopProjectionPileUpPFCandidatesOnPFCandidates_cfi import pfTopProjectionPileUpPFCandidatesOnPFCandidates as tp
pileUpOnPFCandidates = tp.clone()
pileUpOnPFCandidates.name = 'pileUpOnPFCandidates'
pileUpOnPFCandidates.topCollection = 'pfPileUp'
pileUpOnPFCandidates.bottomCollection = 'particleFlow'

pfNoPileUpSequence = cms.Sequence(
    pileUpOnPFCandidates 
    )

