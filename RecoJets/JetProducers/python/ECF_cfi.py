import FWCore.ParameterSet.Config as cms

ECF = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("ECF"),
             Njets = cms.vuint32(1, 2, 3),
             beta = cms.double(1.0),        # CMS default is 1
             )

ECFCbeta1 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("C"),
             Njets = cms.vuint32(1, 2, 3),
             beta = cms.double(1.0),        # CMS default is 1
             )

ECFCbeta2 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("C"),
             Njets = cms.vuint32(1, 2, 3),
             beta = cms.double(2.0),
             )

ECFDbeta1 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("D"),
             Njets = cms.vuint32(2),
             beta = cms.double(1.0),
             )

ECFDbeta2 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("D"),
             Njets = cms.vuint32(2),
             beta = cms.double(2.0),
             )


ECFMbeta1 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("M"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(1.0),
             )

ECFMbeta2 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("M"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(2.0),
             )

ECFNbeta1 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("N"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(1.0),
             )

ECFNbeta2 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("N"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(2.0),
             )

ECFUbeta1 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("U"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(1.0),
             )

ECFUbeta2 = cms.EDProducer("ECFAdder",
             src = cms.InputTag("ak8CHSJets"),
             ecftype = cms.string("U"),
             Njets = cms.vuint32(1,2,3),
             beta = cms.double(2.0),
             )

