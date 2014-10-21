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
metOpt = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("calotoweroptmaker"),
    alias = cms.string('RawCaloMETOpt'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    calculateSignificance = cms.bool(False)
    )

##____________________________________________________________________________||
metOptHO = metOpt.clone()
metOptHO.src = "calotoweroptmakerWithHO"
metOptHO.alias = 'RawCaloMETOptHO'

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
metOptNoHF = cms.EDProducer(
    "CaloMETProducer",
    src = cms.InputTag("calotoweroptmaker"),
    alias = cms.string('RawCaloMETOptNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.0),
    calculateSignificance = cms.bool(False)
    )

##____________________________________________________________________________||
metOptNoHFHO = metOptNoHF.clone()
metOptNoHFHO.src = "calotoweroptmakerWithHO"
metOptNoHFHO.alias = 'RawCaloMETOptNoHFHO'

##____________________________________________________________________________||

