import FWCore.ParameterSet.Config as cms

sistripLACalib = cms.EDFilter("SiStripCalibLorentzAngle",
    Record = cms.string('SiStripLorentzAngleRcd'),
    IOVMode = cms.string('Run'), 
    LayerDB = cms.untracked.bool(False),
    ModuleFitXMin = cms.double(-0.5),
    ModuleFitXMax = cms.double(0.3),    
    ModuleFit2ITXMin = cms.double(-0.4),
    ModuleFit2ITXMax = cms.double(0.2),
    FitCuts_Entries = cms.double(1000),
    FitCuts_p0 = cms.double(10),
    FitCuts_p1 = cms.double(0.3),
    FitCuts_p2 = cms.double(1),
    FitCuts_chi2 = cms.double(10),
    FitCuts_ParErr_p0 = cms.double(0.001),
    p0_guess = cms.double(-0.1),
    p1_guess = cms.double(0.5),
    p2_guess = cms.double(1),
    GaussFitRange = cms.double(0.1),
    DirName = cms.untracked.string('SiStrip'),
    fileName = cms.untracked.string('LorentzAngle.root'),
    out_fileName = cms.untracked.string('LA_plots.root'),
    LA_Report = cms.untracked.string('LA_Report.txt'),
    doStoreOnDB = cms.bool(True),
    SinceAppendMode = cms.bool(True)
)


