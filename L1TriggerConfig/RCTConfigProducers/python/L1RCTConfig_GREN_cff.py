import FWCore.ParameterSet.Config as cms

l1RctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RCTParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

RCTConfigProducers = cms.ESProducer("RCTConfigProducers",
    eGammaHCalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    eMaxForFGCut = cms.double(-999.0), ## to disable it

    noiseVetoHB = cms.bool(False),
    eMaxForHoECut = cms.double(-999.0), ## disabled here

    hOeCut = cms.double(999.0), ## H/E not used

    eGammaECalScaleFactors = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0, 1.0, 1.0, 
        1.0, 1.0, 1.0),
    eMinForHoECut = cms.double(999.0), ## H/E cut not used, 

    hActivityCut = cms.double(999.0), ## not used in GR

    eActivityCut = cms.double(999.0), ## Activity bits for tau calc

    jetMETHCalScaleFactors = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0),
    noiseVetoHEplus = cms.bool(False),
    eicIsolationThreshold = cms.double(0.0), ## force non-isolation

    jetMETLSB = cms.double(1.0),
    # jetMET path not used
    jetMETECalScaleFactors = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0),
    eGammaLSB = cms.double(1.0),
    eMinForFGCut = cms.double(999.0), ## FG cut not used, this serves

    hMinForHoECut = cms.double(999.0), ##

    noiseVetoHEminus = cms.bool(False)
)



