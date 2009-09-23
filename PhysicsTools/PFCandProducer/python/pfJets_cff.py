import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.iterativeCone5PFJets_cff
from PhysicsTools.PFCandProducer.ParticleSelectors.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets


allPfJets = RecoJets.JetProducers.iterativeCone5PFJets_cff.iterativeCone5PFJets.clone()
allPfJets.src = 'pfNoElectron'

pfJets.src = 'allPfJets'
pfJets.ptMin = 10

pfJetSequence = cms.Sequence(
    allPfJets *
    pfJets 
    )
