import FWCore.ParameterSet.Config as cms

ecal_sim_parameter_map = cms.PSet(
    binOfMaximum = cms.int32(6),
    doPhotostatistics = cms.bool(True),
    photoelectronsToAnalogBarrel = cms.double(0.000444444),
    photoelectronsToAnalogEndcap = cms.double(0.000555555),
    readoutFrameSize = cms.int32(10),
    samplingFactor = cms.double(1.0),
    simHitToPhotoelectronsBarrel = cms.double(2250.0),
    simHitToPhotoelectronsEndcap = cms.double(1800.0),
    syncPhase = cms.bool(True),
    timeDependent = cms.bool(False),
    timePhase = cms.double(0.0)
)