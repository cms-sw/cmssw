import FWCore.ParameterSet.Config as cms

#For curious user only
#EGamma POG is recommending use of Towers for HCAL Iso
EleIsoHcalFromHitsExtractorBlock = cms.PSet(
    ComponentName = cms.string('EgammaHcalExtractor'),
    DepositLabel = cms.untracked.string(''),
    hcalRecHits = cms.InputTag("hbhereco"),
    extRadius = cms.double(0.6),
    intRadius = cms.double(0.0),
    etMin = cms.double(-999.0)
)

EleIsoHcalFromTowersExtractorBlock = cms.PSet(
    caloTowers = cms.InputTag('towerMaker'),
    ComponentName = cms.string('EgammaTowerExtractor'),
    intRadius = cms.double(0.0),
    extRadius = cms.double(0.6),
    DepositLabel = cms.untracked.string(''),
    etMin = cms.double(-999.0),
    hcalDepth = cms.int32(-1)
)

