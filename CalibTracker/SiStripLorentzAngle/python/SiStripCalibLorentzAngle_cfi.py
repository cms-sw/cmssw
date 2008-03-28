import FWCore.ParameterSet.Config as cms

sistripLorentzAngle = cms.EDFilter("SiStripCalibLorentzAngle",
    ModuleFitXMax = cms.double(0.3),
    TOBFitXMin = cms.double(-0.3),
    Record = cms.string('SiStripLorentzAngleRcd'),
    ModuleFitXMin = cms.double(-0.3),
    fileName = cms.untracked.string('LorentzAngle.root'),
    #parameters of the base class (ConditionDBWriter)
    IOVMode = cms.string('Run'),
    TOBFitXMax = cms.double(0.3),
    TIBFitXMin = cms.double(-0.3),
    TIBFitXMax = cms.double(0.3),
    doStoreOnDB = cms.bool(True),
    SinceAppendMode = cms.bool(True)
)


