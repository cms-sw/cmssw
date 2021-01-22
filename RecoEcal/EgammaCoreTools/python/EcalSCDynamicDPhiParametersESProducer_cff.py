import FWCore.ParameterSet.Config as cms

ecalSCDynamicDPhiParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('EcalSCDynamicDPhiParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

from RecoEcal.EgammaCoreTools.EcalSCDynamicDPhiParametersESProducer_cfi import ecalSCDynamicDPhiParametersESProducer

