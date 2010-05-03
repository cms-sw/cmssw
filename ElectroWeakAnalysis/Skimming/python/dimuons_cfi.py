import FWCore.ParameterSet.Config as cms

dimuons = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    #  string decay = "goodMuons@+ goodMuons@-"
#    decay = cms.string('selectedLayer1Muons@+ selectedLayer1Muons@-')
    decay = cms.string('selectedPatMuonsTriggerMatch@+ selectedPatMuonsTriggerMatch@-')
)


