import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("CosMuoGenProducer",
    ZCentrOfTarget = cms.double(0.0),
    MinP = cms.double(10.0),
    MinP_CMS = cms.double(-1.0), ##negative means MinP_CMS = MinP. Only change this if you know what you are doing!
    MaxP = cms.double(3000.0),
    MinTheta = cms.double(0.0),
    MaxTheta = cms.double(75.),
    MinPhi = cms.double(0.0),
    MaxPhi = cms.double(360.0),
    MinT0 = cms.double(-12.5),
    MaxT0 = cms.double(12.5),
    PlugVx = cms.double(0.0),
    PlugVz = cms.double(-14000.0),                
    MinEnu = cms.double(10.),                
    MaxEnu = cms.double(10000.),                
    NuProdAlt = cms.double(7.5e6),                       
    AcptAllMu = cms.bool(False), 
    ElossScaleFactor = cms.double(1.0),
    RadiusOfTarget = cms.double(8000.0),
    ZDistOfTarget = cms.double(15000.0),
    TrackerOnly = cms.bool(False),
    TIFOnly_constant = cms.bool(False),
    TIFOnly_linear = cms.bool(False),
    MTCCHalf = cms.bool(False),
    Verbosity = cms.bool(False)
)
