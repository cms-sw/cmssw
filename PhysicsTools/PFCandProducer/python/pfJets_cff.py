import FWCore.ParameterSet.Config as cms

import RecoJets.JetProducers.iterativeCone5PFJets_cff

pfJets = RecoJets.JetProducers.iterativeCone5PFJets_cff.iterativeCone5PFJets.clone()
pfJets.src = 'pfNoMuons:PFCandidates'
