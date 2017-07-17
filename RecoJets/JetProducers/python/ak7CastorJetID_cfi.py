import FWCore.ParameterSet.Config as cms

ak7CastorJetID = cms.EDProducer('CastorJetIDProducer',
        src = cms.InputTag('ak7CastorJets')
)
