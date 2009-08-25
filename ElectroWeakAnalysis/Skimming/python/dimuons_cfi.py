import FWCore.ParameterSet.Config as cms

dimuons = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    #  string decay = "goodMuons@+ goodMuons@-"
    decay = cms.string('selectedLayer1Muons@+ selectedLayer1Muons@-')
#    decay = cms.string('selectedLayer1MuonsTriggerMatch@+ selectedLayer1MuonsTriggerMatch@-')
)


