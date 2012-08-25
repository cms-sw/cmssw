import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.JetIDProducers_cff import ak5JetID

recoJetId = cms.Sequence( ak5JetID )
