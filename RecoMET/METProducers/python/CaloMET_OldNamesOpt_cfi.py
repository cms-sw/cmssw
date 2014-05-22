import FWCore.ParameterSet.Config as cms

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

