import FWCore.ParameterSet.Config as cms

from CalibCalorimetry.CastorCalib.castorDbProducer_cfi import castorDbProducer as _castorDbProducer
CastorDbProducer = _castorDbProducer.clone(
    appendToDataLabel = cms.string( "" )
)
