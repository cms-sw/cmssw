import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.iterativeCone5PFJets_cff
from PhysicsTools.PFCandProducer.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets


allPfJets = RecoJets.JetProducers.iterativeCone5PFJets_cff.iterativeCone5PFJets.clone()
allPfJets.src = 'muonsOnNoPileUp'

pfJets.src = 'allPfJets'
pfJets.ptMin = 10

pfJetSequence = cms.Sequence(
    allPfJets *
    pfJets 
    )
