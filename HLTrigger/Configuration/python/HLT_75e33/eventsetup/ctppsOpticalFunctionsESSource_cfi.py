import FWCore.ParameterSet.Config as cms

ctppsOpticalFunctionsESSource = cms.ESSource("CTPPSOpticalFunctionsESSource",
    appendToDataLabel = cms.string(''),
    configuration = cms.VPSet(cms.PSet(
        opticalFunctions = cms.VPSet(
            cms.PSet(
                fileName = cms.FileInPath('CalibPPS/ESProducers/data/optical_functions/2018/version4/120urad.root'),
                xangle = cms.double(120)
            ),
            cms.PSet(
                fileName = cms.FileInPath('CalibPPS/ESProducers/data/optical_functions/2018/version4/130urad.root'),
                xangle = cms.double(130)
            ),
            cms.PSet(
                fileName = cms.FileInPath('CalibPPS/ESProducers/data/optical_functions/2018/version4/140urad.root'),
                xangle = cms.double(140)
            )
        ),
        validityRange = cms.EventRange(0, 0, 1, 999999, 0, 0)
    )),
    label = cms.string('')
)
