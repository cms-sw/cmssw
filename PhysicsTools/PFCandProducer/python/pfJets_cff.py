import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets
from PhysicsTools.PFCandProducer.Tools.jetTools import jetAlgo


#allPfJets = RecoJets.JetProducers.ic5PFJets_cfi.iterativeCone5PFJets.clone()
allPfJets = jetAlgo('IC5')

pfJets.src = 'allPfJets'
pfJets.ptMin = 10

pfJetSequence = cms.Sequence(
    allPfJets *
    pfJets 
    )
