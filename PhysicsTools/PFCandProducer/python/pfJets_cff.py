import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.ic5PFJets_cfi
from PhysicsTools.PFCandProducer.ParticleSelectors.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets


allPfJets = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
allPfJets.src = 'pfNoElectron'

pfJets.src = 'allPfJets'
pfJets.ptMin = 10

pfJetSequence = cms.Sequence(
    allPfJets *
    pfJets 
    )
