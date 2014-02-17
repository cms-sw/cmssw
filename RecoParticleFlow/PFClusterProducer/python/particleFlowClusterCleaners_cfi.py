import FWCore.ParameterSet.Config as cms

spikeAndDoubleSpikeCleaner_ECAL = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),    
    cleaningByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("ECAL_BARREL"),
                 #single spike
                 singleSpikeThresh = cms.double(4.0),
                 minS4S1_a = cms.double(0.04), #constant term
                 minS4S1_b = cms.double(-0.024), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(10.0),
                 doubleSpikeS6S2 = cms.double(0.04),
                 energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
                 ),
       cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                 #single spike
                 singleSpikeThresh = cms.double(15.0),
                 minS4S1_a = cms.double(0.02), #constant term
                 minS4S1_b = cms.double(-0.0125), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(1e9),
                 doubleSpikeS6S2 = cms.double(-1.0),
                 energyThresholdModifier = cms.double(2.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(3.0) ## aka "tighterF"
                 )
       )
    )

spikeAndDoubleSpikeCleaner_HFEM = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),    
    cleaningByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HF_EM"),
                 #single spike
                 singleSpikeThresh = cms.double(80.0),
                 minS4S1_a = cms.double(0.11), #constant term
                 minS4S1_b = cms.double(-0.19), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(1e9),
                 doubleSpikeS6S2 = cms.double(-1.0),
                 energyThresholdModifier = cms.double(1.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(1.0) ## aka "tighterF"
                 )
       )
    )

spikeAndDoubleSpikeCleaner_HFHAD = cms.PSet(
    algoName = cms.string("SpikeAndDoubleSpikeCleaner"),
    cleaningByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HF_HAD"),
                 #single spike
                 singleSpikeThresh = cms.double(120.0),
                 minS4S1_a = cms.double(0.045), #constant term
                 minS4S1_b = cms.double(-0.080), #log pt scaling
                 #double spike
                 doubleSpikeThresh = cms.double(1e9),
                 doubleSpikeS6S2 = cms.double(-1.0),
                 energyThresholdModifier = cms.double(1.0), ## aka "tighterE"
                 fractionThresholdModifier = cms.double(1.0) ## aka "tighterF"
                 )
       )
    )

rbxAndHPDCleaner = cms.PSet(    
    algoName = cms.string("RBXAndHPDCleaner")
    )
