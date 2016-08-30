import FWCore.ParameterSet.Config as cms

# Configuration parameters for Method 2
m2Parameters = cms.PSet(

    applyPedConstraint    = cms.bool(True),
    applyTimeConstraint   = cms.bool(True),
    applyPulseJitter      = cms.bool(False),
    applyTimeSlew         = cms.bool(True),           #units
    ts4Min                = cms.double(0.),           #fC
    ts4Max                = cms.vdouble(100.,70000.), #fC # this is roughly 20 GeV
    pulseJitter           = cms.double(1.),           #GeV/bin 
    ###
    meanTime              = cms.double(0.),   #ns 
    timeSigmaHPD          = cms.double(5.),   #ns 
    timeSigmaSiPM         = cms.double(2.5),  #ns
    meanPed               = cms.double(0.),   #GeV
    pedSigmaHPD           = cms.double(0.5),  #GeV
    pedSigmaSiPM          = cms.double(1.5),  #GeV # placeholder for siPM
    noiseHPD              = cms.double(1),    #fC
    noiseSiPM             = cms.double(2),    #fC
    ###
    timeMin               = cms.double(-12.5),#ns
    timeMax               = cms.double(12.5), #ns
    ts4chi2               = cms.double(15.),  #chi2 for triple pulse # placeholder for siPM
    fitTimes              = cms.int32(1)      # -1 means no constraint on number of fits per channel

)
