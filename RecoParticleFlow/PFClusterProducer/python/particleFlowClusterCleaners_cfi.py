import FWCore.ParameterSet.Config as cms

spikeAndDoubleSpikeCleaner_EB = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),
    #single spike
    cleaningThreshold = cms.double(4.0),
    minS4S1_a = cms.double(0.04), #constant term
    minS4S1_b = cms.double(-0.024), #log pt scaling
    #double spike
    doubleSpikeThresh = cms.double(10.0),
    doubleSpikeS6S2 = cms.double(0.04),
    energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
    fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
    )

spikeAndDoubleSpikeCleaner_EE = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),
    #single spike
    cleaningThreshold = cms.double(15.0),
    minS4S1_a = cms.double(0.02), #constant term
    minS4S1_b = cms.double(-0.0125), #log pt scaling
    #double spike
    doubleSpikeThresh = cms.double(1e9),
    doubleSpikeS6S2 = cms.double(-1.0),
    energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
    fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
    )
