import FWCore.ParameterSet.Config as cms

hbheprereco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(4),
    tsFromDB = cms.bool(True),
    puCorrMethod = cms.int32(2), 
    
    applyPedConstraint    = cms.bool(True),
    applyTimeConstraint   = cms.bool(True),
    applyPulseJitter      = cms.bool(False),  
    applyUnconstrainedFit = cms.bool(False),   #Turn on original Method 2
    applyTimeSlew         = cms.bool(True),   #units
    ts4Min                = cms.double(5.),   #fC
    ts4Max                = cms.double(500.),   #fC
    pulseJitter           = cms.double(1.),   #GeV/bin
    meanTime              = cms.double(-2.5), #ns
    timeSigma             = cms.double(5.),  #ns
    meanPed               = cms.double(0.),   #GeV
    pedSigma              = cms.double(0.5),  #GeV
    noise                 = cms.double(1),    #fC
    timeMin               = cms.double(-15),  #ns
    timeMax               = cms.double( 10),  #ns
    ts3chi2               = cms.double(5.),   #chi2 (not used)
    ts4chi2               = cms.double(0.),   #chi2 for triple pulse 
    ts345chi2             = cms.double(100.), #chi2 (not used)
    chargeMax             = cms.double(6.),    #Charge cut (fC) for uncstrianed Fit 
    fitTimes              = cms.int32(-1)       # -1 means no constraint on number of fits per channel
)


