import FWCore.ParameterSet.Config as cms

ak8CastorJetID = cms.EDProducer('CastorJetIDProducer',
        src = cms.InputTag('ak8BasicJets')
)
