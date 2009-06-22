import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfTPPileUpPFCandidatesOnPFCandidates_cfi import pfTPPileUpPFCandidatesOnPFCandidates as tp
pileUpOnPFCandidates = tp.clone()
pileUpOnPFCandidates.name = 'pileUpOnPFCandidates'
pileUpOnPFCandidates.topCollection = 'pfPileUp'
pileUpOnPFCandidates.bottomCollection = 'particleFlow'

pfNoPileUpSequence = cms.Sequence(
    pileUpOnPFCandidates 
    )

