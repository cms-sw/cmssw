import FWCore.ParameterSet.Config as cms
import RecoLocalCalo.HcalRecProducers.HBHEMethod3Parameters_cfi as method3
import RecoLocalCalo.HcalRecProducers.HBHEMethod2Parameters_cfi as method2
import RecoLocalCalo.HcalRecProducers.HBHEMethod0Parameters_cfi as method0
import RecoLocalCalo.HcalRecProducers.HBHEPulseShapeFlagSetter_cfi as pulseShapeFlag
import RecoLocalCalo.HcalRecProducers.HBHEStatusBitSetter_cfi as hbheStatusFlag

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

    # include SiPM dark current contribution in pedestal mean
    saveEffectivePedestal = cms.bool(False),

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

    # Parameters which define how we calculate the charge for the basic SiPM
    # nonlinearity correction. To sum up the charge in all time slices
    # (e.g., for cosmics), set sipmQTSShift to -100 and sipmQNTStoSum to 200.
    sipmQTSShift = cms.int32(0),
    sipmQNTStoSum = cms.int32(3),

    # Configure the reconstruction algorithm
    algorithm = cms.PSet(
        # Parameters for "Method 3" (non-keyword arguments have to go first)
        method3.m3Parameters,
        method2.m2Parameters,
        method0.m0Parameters,

        Class = cms.string("SimpleHBHEPhase1Algo"),

        # Time shift (in ns) to add to TDC timing (for QIE11)
        tdcTimeShift = cms.double(0.0),

        # Parameters for "Method 0"
        firstSampleShift = cms.int32(0),

        # Use "Method 2"?
        useM2 = cms.bool(True),

        # Use "Method 3"?
        useM3 = cms.bool(True)
    ),

    # Reconstruction algorithm configuration data to fetch from DB, if any
    algoConfigClass = cms.string(""),

    # Turn rechit status bit setters on/off
    setNegativeFlagsQIE8 = cms.bool(True),
    setNegativeFlagsQIE11 = cms.bool(False),
    setNoiseFlagsQIE8 = cms.bool(True),
    setNoiseFlagsQIE11 = cms.bool(False),
    setPulseShapeFlagsQIE8 = cms.bool(True),
    setPulseShapeFlagsQIE11 = cms.bool(False),
    setLegacyFlagsQIE8 = cms.bool(True),
    setLegacyFlagsQIE11 = cms.bool(False),

    # Parameter sets configuring rechit status bit setters
    flagParametersQIE8 = cms.PSet(
        hbheStatusFlag.qie8Config
    ),
    flagParametersQIE11 = cms.PSet(),

    pulseShapeParametersQIE8 = cms.PSet(
        pulseShapeFlag.qie8Parameters
    ),
    pulseShapeParametersQIE11 = cms.PSet()
)

# Disable the "triangle peak fit" and the corresponding HBHETriangleNoise flag
hbheprereco.pulseShapeParametersQIE8.TrianglePeakTS = cms.uint32(10000)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify(hbheprereco, saveEffectivePedestal = cms.bool(True))
