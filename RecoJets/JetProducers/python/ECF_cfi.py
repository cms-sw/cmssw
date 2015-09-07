import FWCore.ParameterSet.Config as cms

ECF = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             Njets = cms.vuint32(1, 2, 3),
             beta = cms.double(1.0),        # CMS default is 1
             )
