import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

## strip muon content to a minimum
patMuons.embedTrack              = False
patMuons.embedCombinedMuon       = False
patMuons.embedStandAloneMuon     = False
patMuons.embedPickyMuon          = False
patMuons.embedTpfmsMuon          = False
patMuons.embedDytMuon            = False
patMuons.embedPFCandidate        = False
patMuons.embedCaloMETMuonCorrs   = False
patMuons.embedTcMETMuonCorrs     = False
patMuons.embedGenMatch           = False
patMuons.embedHighLevelSelection = False

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import *

## strip jet content to a minimum
patJets.embedCaloTowers          = False
patJets.embedPFCandidates        = False
patJets.addTagInfos              = False
patJets.addAssociatedTracks      = False
patJets.addJetCharge             = False
patJets.addJetID                 = False
patJets.embedGenPartonMatch      = False
patJets.embedGenJetMatch         = False
