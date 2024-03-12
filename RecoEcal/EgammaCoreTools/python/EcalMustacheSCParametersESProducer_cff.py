import FWCore.ParameterSet.Config as cms

ecalMustacheSCParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('EcalMustacheSCParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

from RecoEcal.EgammaCoreTools.EcalMustacheSCParametersESProducer_cfi import ecalMustacheSCParametersESProducer

# foo bar baz
# RGzP0D5I09a1x
