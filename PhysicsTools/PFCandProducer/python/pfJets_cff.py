import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.iterativeCone5PFJets_cff
from PhysicsTools.PFCandProducer.pfNoMuonsNoPileUp_cfi import *
from PhysicsTools.PFCandProducer.ptMinPFJetSelector_cfi import ptMinPFJets as pfJets


allPfJets = RecoJets.JetProducers.iterativeCone5PFJets_cff.iterativeCone5PFJets.clone()
allPfJets.src = 'pfNoMuonsNoPileUp:PFCandidates'

pfJets.src = 'allPfJets'


pfJetSequence = cms.Sequence(
    pfNoMuonsNoPileUp+
    allPfJets+
    pfJets
    )
