import FWCore.ParameterSet.Config as cms

RCTConfigProducers = cms.ESProducer("RCTConfigProducers",
    eGammaHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    eMaxForFGCut = cms.double(50.0),
    noiseVetoHB = cms.bool(False),
    eMaxForHoECut = cms.double(60.0),
    hOeCut = cms.double(0.05),
    eGammaECalScaleFactors = cms.vdouble(1.0, 1.01, 1.02, 1.02, 1.02, 1.06, 1.04, 1.04, 1.05, 1.09, 1.1, 1.1, 1.15, 1.2, 1.27, 1.29, 1.32, 1.52, 1.52, 1.48, 1.4, 1.32, 1.26, 1.21, 1.17, 1.15, 1.15, 1.15),
    eMinForHoECut = cms.double(3.0),
    hActivityCut = cms.double(3.0),
    eActivityCut = cms.double(3.0),
    jetMETHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    noiseVetoHEplus = cms.bool(False),
    eicIsolationThreshold = cms.double(3.0),
    jetMETLSB = cms.double(0.5),
    jetMETECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    eGammaLSB = cms.double(0.5),
    eMinForFGCut = cms.double(3.0),
    hMinForHoECut = cms.double(3.0),
    noiseVetoHEminus = cms.bool(False)
)


