import FWCore.ParameterSet.Config as cms

source = cms.Source("CosMuoGenSource",
    RadiusOfTarget = cms.double(8000.0),
    TIFOnly_constant = cms.bool(False),
    TIFOnly_linear = cms.bool(False),
    ElossScaleFactor = cms.double(1.0),
    MaxPhi = cms.double(360.0),
    MaxTheta = cms.double(84.26),
    MaxT0 = cms.double(12.5),
    ZDistOfTarget = cms.double(15000.0),
    MinP = cms.double(3.0),
    MinT0 = cms.double(-12.5),
    MTCCHalf = cms.bool(False),
    TrackerOnly = cms.bool(False),
    MinTheta = cms.double(0.0),
    MinP_CMS = cms.double(-1.0), ##negative means MinP_CMS = MinP. Only change this if you know what you are doing!

    MaxP = cms.double(3000.0),
    MinPhi = cms.double(0.0),
    Verbosity = cms.bool(False)
)


