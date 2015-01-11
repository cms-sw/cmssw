import FWCore.ParameterSet.Config as cms

hbheUpgradeReco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(5.0),  
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis","HBHEUpgradeDigiCollection"),
    Subdetector = cms.string('upgradeHBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(2),
    tsFromDB = cms.bool(True), 
    firstDepthWeight = cms.double(0.417),  # 0.5/1.2
    puCorrMethod = cms.int32(2), 
    
    applyPedConstraint    = cms.bool(True),
    applyTimeConstraint   = cms.bool(False),
    applyPulseJitter      = cms.bool(False),  
    applyUnconstrainedFit = cms.bool(False),   #Turn on original Method 2
    applyTimeSlew         = cms.bool(False),   #units
    ts4Min                = cms.double(100.),   #fC
    ts4Max                = cms.double(10000000.),   #fC # this value should be irrelevant & removed from If statements in slhc 
    pulseJitter           = cms.double(1.),   #GeV/bin
    meanTime              = cms.double(5.), #ns
    timeSigma             = cms.double(5.),  #ns
    meanPed               = cms.double(0.),   #GeV
    pedSigma              = cms.double(0.5),  #GeV
    noise                 = cms.double(1),    #fC
    timeMin               = cms.double(-7.5),
    timeMax               = cms.double(17.5),  #ns
    ts3chi2               = cms.double(5.),   #chi2 (not used)
    ts4chi2               = cms.double(15.),   #chi2 for triple pulse 
    ts345chi2             = cms.double(100.), #chi2 (not used)
    chargeMax             = cms.double(6.),    #Charge cut (fC) for uncstrianed Fit 
    fitTimes              = cms.int32(-1)       # -1 means no constraint on number of fits per channel
)

