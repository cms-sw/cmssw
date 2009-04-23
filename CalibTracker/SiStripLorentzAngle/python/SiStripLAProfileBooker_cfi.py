# The following comments couldn't be translated into the new config version:

# input from trajectories

# calibration

import FWCore.ParameterSet.Config as cms

sistripLorentzAngle = cms.EDFilter("SiStripLAProfileBooker",
    ModuleXMax = cms.double(0.6),
    treeName = cms.untracked.string('SiStripLATrees.root'),
    #output files
    fitName = cms.string('fit_results'),
    ModuleFitXMax = cms.double(0.3),
    TOBFitXMin = cms.double(-0.3),
    TOBXMin = cms.double(-0.6),
    TIBXMax = cms.double(0.6),
    Tracks = cms.InputTag("generalTracks"),
    ModuleXMin = cms.double(-0.6),
    fileName = cms.untracked.string('trackhisto.root'),
    ModuleFitXMin = cms.double(-0.3),
    TIBXMin = cms.double(-0.6),
    TIBFitXMin = cms.double(-0.3),
    UseStripCablingDB = cms.bool(False),
    TIBFitXMax = cms.double(0.3),
    NHitMin = cms.int32(8),
    TOBFitXMax = cms.double(0.3),
    DoCalibration = cms.bool(False),
    TOBXMax = cms.double(0.6)
)


