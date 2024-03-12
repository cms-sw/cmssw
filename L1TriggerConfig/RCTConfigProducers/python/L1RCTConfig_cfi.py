import FWCore.ParameterSet.Config as cms
from L1TriggerConfig.RCTConfigProducers.RCTCalibration_cff import *

RCTConfigProducers = cms.ESProducer("RCTConfigProducers",
    rct_calibration,
    eGammaHCalScaleFactors = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.0),
    eMaxForFGCut = cms.double(-999),
    noiseVetoHB = cms.bool(False),
    eMaxForHoECut = cms.double(-999),
    hOeCut = cms.double(0.05),
    eGammaECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0),
    eMinForHoECut = cms.double(999),
    jscQuietThresholdBarrel = cms.uint32(3),
    hActivityCut = cms.double(4.0),
    eActivityCut = cms.double(4.0),
    jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0),
    noiseVetoHEplus = cms.bool(False),
    eicIsolationThreshold = cms.uint32(7),
    jetMETLSB = cms.double(0.25),
    jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0, 1.0, 1.0, 
                                         1.0, 1.0, 1.0),
    eMinForFGCut = cms.double(999),
    eGammaLSB = cms.double(0.25),
    jscQuietThresholdEndcap = cms.uint32(3),
    hMinForHoECut = cms.double(3.0),
    noiseVetoHEminus = cms.bool(False),
    useCorrectionsLindsey = cms.bool(False)                                
)



# foo bar baz
# asVzUOGLAnUGa
# yWbKfteh6ZAP6
