import FWCore.ParameterSet.Config as cms


SiStripBackPlaneCorrectionGenerator = cms.Service(
    "SiStripBackPlaneCorrectionGenerator",
  
    #BackPlaneCorrection values for each module geometry: IB1, IB2, OB1, OB2, W1A, W1B, W2A, W2B, W3A, W3B, W4, W5, W6, W7
    BackPlaneCorrection_PerModuleGeometry = cms.vdouble(0.034, 0.034, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),         
)
