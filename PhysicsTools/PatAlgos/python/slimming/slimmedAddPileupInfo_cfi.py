import FWCore.ParameterSet.Config as cms

slimmedAddPileupInfo = cms.EDProducer(
    'PileupSummaryInfoSlimmer',
    src = cms.InputTag('addPileupInfo'),
    keepDetailedInfoFor = cms.vint32(0)
)
# foo bar baz
# 5O6vGOlt6buq9
# 9nvCaVKhp1epw
