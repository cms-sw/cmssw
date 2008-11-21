import FWCore.ParameterSet.Config as cms

common = cms.PSet(
    CalibrationInputs = cms.VInputTag(),
    EcalTPGInput = cms.InputTag("ecalTriggerPrimitiveDigis"),
    HcalTPGInput = cms.InputTag("hcalTriggerPrimitiveDigis"),
    RegionsInput = cms.InputTag("rctDigis"),
    OutputFile = cms.string("RCTCalibration"),
    DeltaEtaBarrel = cms.double(0.0870),
    TowersInBarrel = cms.int32(20),
    TowerDeltaPhi = cms.double(0.0870),
    EndcapEtaBoundaries = cms.vdouble( 0.09, 0.1, 0.113, 0.129, 0.15, 0.178, 0.15, 0.35 ),
    PythonOut = cms.untracked.bool(True),
    debug = cms.untracked.int32(-1),
    FarmoutMode = cms.untracked.bool(False)
    )
