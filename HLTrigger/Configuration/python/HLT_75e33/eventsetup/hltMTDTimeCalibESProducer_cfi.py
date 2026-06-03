import FWCore.ParameterSet.Config as cms

def _addProcessMTDTimeCalibESProducer(process):
    from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import mtdDigitizer as _mtdDigitizer
    process.hltMTDTimeCalibESProducer = cms.ESProducer('MTDTimeCalibESProducer',
                                                       BTLLightCollSlope = _mtdDigitizer.barrelDigitizer.DeviceSimulation.LightCollectionSlope,
                                                       appendToDataLabel = cms.string(''))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForMTDTimeCalibESProducer_ = mtd_at_hlt.makeProcessModifier(_addProcessMTDTimeCalibESProducer)
