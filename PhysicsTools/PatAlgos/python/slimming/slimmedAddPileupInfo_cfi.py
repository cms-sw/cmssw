import FWCore.ParameterSet.Config as cms

slimmedAddPileupInfo = cms.EDProducer(
    'PileupSummaryInfoSlimmer',
    src = cms.InputTag('addPileupInfo'),
    keepDetailedInfoFor = cms.vint32(0)
)
