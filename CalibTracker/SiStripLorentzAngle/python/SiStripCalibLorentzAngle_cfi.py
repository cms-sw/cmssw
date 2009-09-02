import FWCore.ParameterSet.Config as cms

sistripLACalib = cms.EDFilter("SiStripCalibLorentzAngle",
    Record = cms.string('SiStripLorentzAngleRcd'),
    IOVMode = cms.string('Run'), 
    ModuleFitXMax = cms.double(0.3),
    ModuleFitXMin = cms.double(-0.3),
    TOBFitXMin = cms.double(-0.3),
    TOBFitXMax = cms.double(0.3),
    TIBFitXMin = cms.double(-0.3),
    TIBFitXMax = cms.double(0.3),
    
    ModuleFit2ITXMin = cms.double(-0.4),
    ModuleFit2ITXMax = cms.double(0.1),
    FitCuts_Entries = cms.double(1000),
    FitCuts_p0 = cms.double(0),
    FitCuts_p1 = cms.double(0.1),
    FitCuts_p2 = cms.double(1),
    FitCuts_chi2 = cms.double(30),
    FitCuts_ParErr_p0 = cms.double(0.001),
    p0_guess = cms.double(-0.1),
    p1_guess = cms.double(0.1),
    p2_guess = cms.double(1),
    fileName = cms.untracked.string('LorentzAngle.root'),
    out_fileName = cms.untracked.string('LA_plots.root'),
    LA_Report = cms.untracked.string('LA_Report.txt'),
    LA_ProbFit = cms.untracked.string('LA_ProbFit.txt'),
    treeName = cms.untracked.string('ModuleTree.root'),
    doStoreOnDB = cms.bool(True),
    SinceAppendMode = cms.bool(True)
)


