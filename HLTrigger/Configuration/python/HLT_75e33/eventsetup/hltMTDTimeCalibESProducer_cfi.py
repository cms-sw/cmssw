import FWCore.ParameterSet.Config as cms

from SimFastTiming.FastTimingCommon.mtdDigitizer_cfi import mtdDigitizer as _mtdDigitizer

hltMTDTimeCalibESProducer = cms.ESProducer('MTDTimeCalibESProducer',
                                           BTLLightCollSlope = _mtdDigitizer.barrelDigitizer.DeviceSimulation.LightCollectionSlope,
                                           appendToDataLabel = cms.string('')
                                           )
