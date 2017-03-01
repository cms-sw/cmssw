import FWCore.ParameterSet.Config as cms

ak5CastorJetID = cms.EDProducer('CastorJetIDProducer',
        src = cms.InputTag('ak5CastorJets')
)
