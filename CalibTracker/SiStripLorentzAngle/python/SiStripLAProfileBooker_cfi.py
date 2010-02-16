# The following comments couldn't be translated into the new config version:

# calibration

import FWCore.ParameterSet.Config as cms

sistripLAProfile = cms.EDAnalyzer("SiStripLAProfileBooker",

    # input from trajectories
    Tracks = cms.InputTag("generalTracks"),
    UseStripCablingDB = cms.bool(False),
    
    # calibration
    DoCalibration = cms.bool(False),
    TIB_bin = cms.int32(30),
    TOB_bin = cms.int32(30),
    SUM_bin = cms.int32(30),
    ModuleXMin = cms.double(-0.6),
    ModuleXMax = cms.double(0.6),
    TIBXMin = cms.double(-0.6),
    TIBXMax = cms.double(0.6),
    TOBXMin = cms.double(-0.6),
    TOBXMax = cms.double(0.6),
    ModuleFitXMin = cms.double(-0.3),
    ModuleFitXMax = cms.double(0.3),
    TIBFitXMin = cms.double(-0.3),
    TIBFitXMax = cms.double(0.3),
    TOBFitXMin = cms.double(-0.3),
    TOBFitXMax = cms.double(0.3),
    #NHitMin = cms.int32(8),
     
    #output files
    treeName = cms.untracked.string('SiStripLATrees.root'),
    fileName = cms.untracked.string('trackhisto.root')
)


