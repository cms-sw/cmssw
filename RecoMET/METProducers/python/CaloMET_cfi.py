import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
caloMet = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("towerMaker"),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.3),
    calculateSignificance = cms.bool(False)
    )

##____________________________________________________________________________||
caloMetBEFO = caloMet.clone()
caloMetBEFO.src = "towerMakerWithHO"
caloMetBEFO.alias = 'RawCaloMETHO'

##____________________________________________________________________________||
caloMetBE = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("towerMaker"),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.3),
    calculateSignificance = cms.bool(False)
)

##____________________________________________________________________________||
caloMetBEO = caloMetBE.clone()
caloMetBEO.src = "towerMakerWithHO"
caloMetBEO.alias = 'RawCaloMETNoHFHO'

##____________________________________________________________________________||
