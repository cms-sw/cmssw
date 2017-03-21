import FWCore.ParameterSet.Config as cms

# customisation functions for the HLT configuration
from HLTrigger.Configuration.common import *

# import the relevant eras from Configuration.Eras.*
from Configuration.Eras.Modifier_HLT_2016_cff import HLT_2016


# modify the HLT configuration for the 2016 HF readout
def customizeHLTfor2016_HF(process):
    if 'hltHfprereco' in process.__dict__:
        # remove hltHfprereco from all sequences and from the process
        delete_modules(process, process.hltHfprereco)

    if 'hltHfreco' in process.__dict__:
        # use the 2016 HF HcalHitReconstructor
        process.hltHfreco = cms.EDProducer("HcalHitReconstructor",
            HFInWindowStat = cms.PSet(
                hflongEthresh = cms.double(40.0),
                hflongMaxWindowTime = cms.vdouble(10.0),
                hflongMinWindowTime = cms.vdouble(-10.0),
                hfshortEthresh = cms.double(40.0),
                hfshortMaxWindowTime = cms.vdouble(10.0),
                hfshortMinWindowTime = cms.vdouble(-12.0) ),
            PETstat = cms.PSet(
                HcalAcceptSeverityLevel = cms.int32(9),
                flagsToSkip = cms.int32(0),
                longETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                longEnergyParams = cms.vdouble(43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62),
                long_R = cms.vdouble(0.98),
                long_R_29 = cms.vdouble(0.8),
                shortETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                shortEnergyParams = cms.vdouble(35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093),
                short_R = cms.vdouble(0.8),
                short_R_29 = cms.vdouble(0.8) ),
            S8S1stat = cms.PSet(
                HcalAcceptSeverityLevel = cms.int32(9),
                flagsToSkip = cms.int32(16),
                isS8S1 = cms.bool(True),
                longETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                longEnergyParams = cms.vdouble(40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                long_optimumSlope = cms.vdouble(0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
                shortETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                shortEnergyParams = cms.vdouble(40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                short_optimumSlope = cms.vdouble(0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1) ),
            S9S1stat = cms.PSet(
                HcalAcceptSeverityLevel = cms.int32(9),
                flagsToSkip = cms.int32(24),
                isS8S1 = cms.bool(False),
                longETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                longEnergyParams = cms.vdouble(43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62),
                long_optimumSlope = cms.vdouble(-99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927),
                shortETParams = cms.vdouble(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                shortEnergyParams = cms.vdouble(35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093),
                short_optimumSlope = cms.vdouble(-99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927) ),
            Subdetector = cms.string('HF'),
            applyPedConstraint = cms.bool(True),
            applyPulseJitter = cms.bool(False),
            applyTimeConstraint = cms.bool(True),
            applyTimeSlew = cms.bool(True),
            applyTimeSlewM3 = cms.bool(True),
            correctForPhaseContainment = cms.bool(False),
            correctForTimeslew = cms.bool(False),
            correctTiming = cms.bool(False),
            correctionPhaseNS = cms.double(13.0),
            dataOOTCorrectionCategory = cms.string('Data'),
            dataOOTCorrectionName = cms.string(''),
            digiLabel = cms.InputTag("hltHcalDigis"),
            digiTimeFromDB = cms.bool(True),
            digistat = cms.PSet(
                HFdigiflagCoef = cms.vdouble(0.93, -0.012667, -0.38275),
                HFdigiflagExpectedPeak = cms.int32(2),
                HFdigiflagFirstSample = cms.int32(1),
                HFdigiflagMinEthreshold = cms.double(40.0),
                HFdigiflagSamplesToAdd = cms.int32(3) ),
            dropZSmarkedPassed = cms.bool(True),
            firstAuxTS = cms.int32(1),
            firstSample = cms.int32(2),
            fitTimes = cms.int32(1),
            flagParameters = cms.PSet( ),
            hfTimingTrustParameters = cms.PSet(
                hfTimingTrustLevel1 = cms.int32(1),
                hfTimingTrustLevel2 = cms.int32(4) ),
            hscpParameters = cms.PSet( ),
            mcOOTCorrectionCategory = cms.string('MC'),
            mcOOTCorrectionName = cms.string(''),
            meanPed = cms.double(0.0),
            meanTime = cms.double(-2.5),
            noiseHPD = cms.double(1.0),
            noiseSiPM = cms.double(1.0),
            pedSigmaHPD = cms.double(0.5),
            pedSigmaSiPM = cms.double(0.00065),
            pedestalUpperLimit = cms.double(2.7),
            puCorrMethod = cms.int32(0),
            pulseJitter = cms.double(1.0),
            pulseShapeParameters = cms.PSet( ),
            recoParamsFromDB = cms.bool(True),
            respCorrM3 = cms.double(0.95),
            samplesToAdd = cms.int32(2),
            saturationParameters = cms.PSet(
                maxADCvalue = cms.int32(127) ),
            setHSCPFlags = cms.bool(False),
            setNegativeFlags = cms.bool(False),
            setNoiseFlags = cms.bool(True),
            setPulseShapeFlags = cms.bool(False),
            setSaturationFlags = cms.bool(False),
            setTimingShapedCutsFlags = cms.bool(False),
            setTimingTrustFlags = cms.bool(False),
            timeMax = cms.double(10.0),
            timeMin = cms.double(-15.0),
            timeSigmaHPD = cms.double(5.0),
            timeSigmaSiPM = cms.double(2.5),
            timeSlewPars = cms.vdouble(12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0),
            timeSlewParsType = cms.int32(3),
            timingshapedcutsParameters = cms.PSet( ),
            ts4Max = cms.vdouble(100.0, 45000.0),
            ts4Min = cms.double(5.0),
            ts4chi2 = cms.vdouble(15.0, 15.0),
            tsFromDB = cms.bool(True),
            useLeakCorrection = cms.bool(False)
        )

    return process


# modify the HLT configuration for the 2016 Pixel detector
def customizeHLTfor2016_Pixel(process):
    return process


# modify the HLT configuration for the 2016 configuration
def customizeHLTfor2016(process):
    process = customizeHLTfor2016_HF(process)
    process = customizeHLTfor2016_Pixel(process)
    return process


# attach `customizeHLTfor2016' to the `HLT_2016' modifier
def modifyHLTfor2016(process):
    HLT_2016.toModify(process, customizeHLTfor2016)

