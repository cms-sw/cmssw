import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
met = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("towerMaker"),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.3),
    calculateSignificance = cms.bool(False)
    )

##____________________________________________________________________________||
metHO = met.clone()
metHO.src = "towerMakerWithHO"
metHO.alias = 'RawCaloMETHO'

##____________________________________________________________________________||
metNoHF = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("towerMaker"),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.3),
    calculateSignificance = cms.bool(False)
)

##____________________________________________________________________________||
metNoHFHO = metNoHF.clone()
metNoHFHO.src = "towerMakerWithHO"
metNoHFHO.alias = 'RawCaloMETNoHFHO'

##____________________________________________________________________________||

