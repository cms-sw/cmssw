import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3

hbheprereco = cms.EDProducer(
    "HBHEPhase1Reconstructor",

    # Label for the input HBHEDigiCollection, and flag indicating
    # whether we should process this collection
    digiLabelQIE8 = cms.InputTag("hcalDigis"),
    processQIE8 = cms.bool(True),

    # Label for the input QIE11DigiCollection, and flag indicating
    # whether we should process this collection
    digiLabelQIE11 = cms.InputTag("hcalDigis"),
    processQIE11 = cms.bool(True),

    # Get the "sample of interest" index from DB?
    # If not, it is taken from the dataframe.
    tsFromDB = cms.bool(False),

    # Use the HcalRecoParam structure from DB inside
    # the reconstruction algorithm?
    recoParamsFromDB = cms.bool(True),

    # Drop zero-suppressed channels?
    dropZSmarkedPassed = cms.bool(True),

    # Flag indicating whether we should produce HBHERecHitCollection
    makeRecHits = cms.bool(True),

    # Flag indicating whether we should produce HBHEChannelInfoCollection
    saveInfos = cms.bool(False),

    # Flag indicating whether we should include HBHEChannelInfo objects
    # into HBHEChannelInfoCollection despite the fact that the channels
    # are either tagged bad in DB of zero-suppressed. Note that the rechit
    # collection will not include such channels even if this flag is set.
    saveDroppedInfos = cms.bool(False),

    # Configure the reconstruction algorithm
    algorithm = cms.PSet(
        # Parameters for "Method 3" (non-keyword arguments have to go first)
        method3.m3Parameters,

        Class = cms.string("SimpleHBHEPhase1Algo"),

        # Time shift (in ns) to add to TDC timing (for QIE11)
        tdcTimeShift = cms.double(0.0),

        # Parameters for "Method 0"
        firstSampleShift  = cms.int32(0),
        samplesToAdd      = cms.int32(2),
        correctionPhaseNS = cms.double(6.0),

        # Use "Method 2"? Change this to True when implemented.
        useM2 = cms.bool(False),

        # Parameters for "Method 2"
        applyPedConstraint    = cms.bool(True),
        applyTimeConstraint   = cms.bool(True),
        applyPulseJitter      = cms.bool(False),  
        applyTimeSlew         = cms.bool(True),   #units
        ts4Min                = cms.double(0.),   #fC
        ts4Max                = cms.vdouble(100.,70000.), #fC # this is roughly 20 GeV
        pulseJitter           = cms.double(1.),   #GeV/bin
        ###
        meanTime              = cms.double(0.),   #ns
        timeSigmaHPD          = cms.double(5.),   #ns
        timeSigmaSiPM         = cms.double(2.5),  #ns
        meanPed               = cms.double(0.),   #GeV
        pedSigmaHPD           = cms.double(0.5),  #GeV
        pedSigmaSiPM          = cms.double(1.5),  #GeV
        noiseHPD              = cms.double(1),    #fC
        noiseSiPM             = cms.double(2),    #fC
        ###
        timeMin               = cms.double(-12.5),#ns
        timeMax               = cms.double(12.5), #ns
        ts4chi2               = cms.double(15.),  #chi2 for triple pulse 
        fitTimes              = cms.int32(1),     # -1 means no constraint on number of fits per channel

        # Use "Method 3"? Change this to True when implemented.
        useM3 = cms.bool(False)
    ),

    # Reconstruction algorithm configuration data to fetch from DB, if any
    algoConfigClass = cms.string("")
)
