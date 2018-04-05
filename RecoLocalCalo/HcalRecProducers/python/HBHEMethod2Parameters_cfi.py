import FWCore.ParameterSet.Config as cms

# Configuration parameters for Method 2
m2Parameters = cms.PSet(

    applyPedConstraint    = cms.bool(True),
    applyTimeConstraint   = cms.bool(True),
    applyPulseJitter      = cms.bool(False),
    applyTimeSlew         = cms.bool(True),           #units
    ts4Min                = cms.double(0.),           #fC
    ts4Max                = cms.vdouble(100., 20000., 30000), #fC # this is roughly 20 GeV, HPD, siPMdepth1, siPMdepth>1
    pulseJitter           = cms.double(1.),           #GeV/bin 
    ###
    meanPed               = cms.double(0.),   #GeV
    meanTime              = cms.double(0.),   #ns 
    timeSigmaHPD          = cms.double(5.),   #ns 
    timeSigmaSiPM         = cms.double(2.5),  #ns
    ###
    timeMin               = cms.double(-12.5),#ns
    timeMax               = cms.double(12.5), #ns
    ts4chi2               = cms.vdouble(15.,15.),  #chi2 for triple pulse
    fitTimes              = cms.int32(1)      # -1 means no constraint on number of fits per channel

)
