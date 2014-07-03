### AUTO-GENERATED CMSRUN CONFIGURATION FOR ECAL DQM ###
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('runkey', default = 'pp_run', mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.string, info = 'Run Keys of CMS')
options.register('runNumber', default = 194533, mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.int, info = "Run number.")
options.register('runInputDir', default = '/fff/BU0/test', mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.string, info = "Directory where the DQM files will appear.")
options.register('skipFirstLumis', default = False, mult = VarParsing.multiplicity.singleton, mytype = VarParsing.varType.bool, info = "Skip (and ignore the minEventsPerLumi parameter) for the files which have been available at the begining of the processing.")

options.parseArguments()


from DQM.Integration.test.dqmPythonTypes import *
runType = RunType(['pp_run','cosmic_run','hi_run','hpu_run'])
if not options.runkey.strip():
    options.runkey = 'pp_run'

runType.setRunType(options.runkey.strip())

import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.source = cms.Source("DQMStreamerReader",
    streamLabel = cms.untracked.string(''),
    delayMillis = cms.untracked.uint32(500),
    runNumber = cms.untracked.uint32(0),
    endOfRunKills = cms.untracked.bool(True),
    runInputDir = cms.untracked.string(''),
    minEventsPerLumi = cms.untracked.int32(1),
    deleteDatFiles = cms.untracked.bool(False),
    skipFirstLumis = cms.untracked.bool(False)
)
process.ecalDetIdToBeRecovered = cms.EDProducer("EcalDetIdToBeRecoveredProducer",
    ebIntegrityChIdErrors = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    ebDetIdToBeRecovered = cms.string('ebDetId'),
    integrityTTIdErrors = cms.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    eeIntegrityGainErrors = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    ebFEToBeRecovered = cms.string('ebFE'),
    ebIntegrityGainErrors = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    eeDetIdToBeRecovered = cms.string('eeDetId'),
    eeIntegrityGainSwitchErrors = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    eeIntegrityChIdErrors = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    ebIntegrityGainSwitchErrors = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    ebSrFlagCollection = cms.InputTag("ecalDigis"),
    eeSrFlagCollection = cms.InputTag("ecalDigis"),
    integrityBlockSizeErrors = cms.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    eeFEToBeRecovered = cms.string('eeFE')
)


process.ecalDigis = cms.EDProducer("EcalRawToDigi",
    tccUnpacking = cms.bool(True),
    FedLabel = cms.InputTag("listfeds"),
    srpUnpacking = cms.bool(True),
    syncCheck = cms.bool(True),
    feIdCheck = cms.bool(True),
    silentMode = cms.untracked.bool(True),
    InputLabel = cms.InputTag("rawDataCollector"),
    orderedFedList = cms.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.bool(True),
    numbTriggerTSamples = cms.int32(1),
    numbXtalTSamples = cms.int32(10),
    orderedDCCIdList = cms.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FEDs = cms.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    DoRegional = cms.bool(False),
    feUnpacking = cms.bool(True),
    forceToKeepFRData = cms.bool(False),
    headerUnpacking = cms.bool(True),
    memUnpacking = cms.bool(True)
)


process.ecalGlobalUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    eePulseShape = cms.vdouble(5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575, 
        1.0, 0.8876, 0.6732, 0.4741, 0.3194),
    EBtimeFitParameters = cms.vdouble(-2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 
        91.01147, -50.35761, 11.05621),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    outOfTimeThresholdGain61pEB = cms.double(5),
    EEtimeNconst = cms.double(31.8),
    EBtimeConstantTerm = cms.double(0.6),
    outOfTimeThresholdGain61pEE = cms.double(10),
    EEamplitudeFitParameters = cms.vdouble(1.89, 1.4),
    EBtimeNconst = cms.double(28.5),
    kPoorRecoFlagEB = cms.bool(True),
    ebPulseShape = cms.vdouble(5.2e-05, -5.26e-05, 6.66e-05, 0.1168, 0.7575, 
        1.0, 0.8876, 0.6732, 0.4741, 0.3194),
    EBtimeFitLimits_Lower = cms.double(0.2),
    kPoorRecoFlagEE = cms.bool(False),
    chi2ThreshEB_ = cms.double(36.0),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEtimeFitParameters = cms.vdouble(-2.390548, 3.553628, -17.62341, 67.67538, -133.213, 
        140.7432, -75.41106, 16.20277),
    EBchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
    EEchi2Parameters = cms.vdouble(2.122, 0.022, 2.122, 0.022),
    outOfTimeThresholdGain12mEE = cms.double(10),
    outOfTimeThresholdGain12mEB = cms.double(5),
    EEtimeFitLimits_Upper = cms.double(1.4),
    EEtimeFitLimits_Lower = cms.double(0.2),
    ebSpikeThreshold = cms.double(1.042),
    EBamplitudeFitParameters = cms.vdouble(1.138, 1.652),
    amplitudeThresholdEB = cms.double(10),
    outOfTimeThresholdGain12pEE = cms.double(10),
    outOfTimeThresholdGain12pEB = cms.double(5),
    amplitudeThresholdEE = cms.double(10),
    outOfTimeThresholdGain61mEB = cms.double(5),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
    outOfTimeThresholdGain61mEE = cms.double(10),
    EEtimeConstantTerm = cms.double(1.0),
    algo = cms.string('EcalUncalibRecHitWorkerGlobal'),
    chi2ThreshEE_ = cms.double(95.0),
    EBtimeFitLimits_Upper = cms.double(1.4)
)


process.ecalLaserLedUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    alphaEB = cms.double(1.138),
    alphaEE = cms.double(1.89),
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    AlphaBetaFilename = cms.untracked.string('NOFILE'),
    betaEB = cms.double(1.655),
    MinAmplEndcap = cms.double(16.0),
    MinAmplBarrel = cms.double(12.0),
    algo = cms.string('EcalUncalibRecHitWorkerFixedAlphaBetaFit'),
    betaEE = cms.double(1.4),
    UseDynamicPedestal = cms.bool(True),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)


process.ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    recoverEEVFE = cms.bool(False),
    EErechitCollection = cms.string('EcalRecHitsEE'),
    recoverEBIsolatedChannels = cms.bool(False),
    recoverEBVFE = cms.bool(False),
    laserCorrection = cms.bool(True),
    EBLaserMIN = cms.double(0.5),
    killDeadChannels = cms.bool(True),
    dbStatusToBeExcludedEB = cms.vint32(14, 78, 142),
    EEuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    dbStatusToBeExcludedEE = cms.vint32(14, 78, 142),
    EELaserMIN = cms.double(0.5),
    ebFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered","ebFE"),
    cleaningConfig = cms.PSet(
        e6e2thresh = cms.double(0.04),
        tightenCrack_e6e2_double = cms.double(3),
        tightenCrack_e4e1_single = cms.double(3),
        cThreshold_barrel = cms.double(4),
        e4e1Threshold_barrel = cms.double(0.08),
        tightenCrack_e1_single = cms.double(2),
        e4e1_b_barrel = cms.double(-0.024),
        e4e1_a_barrel = cms.double(0.04),
        cThreshold_double = cms.double(10),
        ignoreOutOfTimeThresh = cms.double(1000000000.0),
        cThreshold_endcap = cms.double(15),
        e4e1_a_endcap = cms.double(0.02),
        e4e1_b_endcap = cms.double(-0.0125),
        e4e1Threshold_endcap = cms.double(0.3),
        tightenCrack_e1_double = cms.double(2)
    ),
    logWarningEtThreshold_EE_FE = cms.double(50),
    eeDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered","eeDetId"),
    recoverEBFE = cms.bool(True),
    algo = cms.string('EcalRecHitWorkerSimple'),
    ebDetIdToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered","ebDetId"),
    singleChannelRecoveryThreshold = cms.double(8),
    ChannelStatusToBeExcluded = cms.vstring('kNoisy', 
        'kNNoisy', 
        'kFixedG6', 
        'kFixedG1', 
        'kFixedG0', 
        'kNonRespondingIsolated', 
        'kDeadVFE', 
        'kDeadFE', 
        'kNoDataNoTP'),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    singleChannelRecoveryMethod = cms.string('NeuralNetworks'),
    recoverEEFE = cms.bool(True),
    triggerPrimitiveDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EBLaserMAX = cms.double(3.0),
    flagsMapDBReco = cms.PSet(
        kGood = cms.vstring('kOk', 
            'kDAC', 
            'kNoLaser', 
            'kNoisy'),
        kNeighboursRecovered = cms.vstring('kFixedG0', 
            'kNonRespondingIsolated', 
            'kDeadVFE'),
        kDead = cms.vstring('kNoDataNoTP'),
        kNoisy = cms.vstring('kNNoisy', 
            'kFixedG6', 
            'kFixedG1'),
        kTowerRecovered = cms.vstring('kDeadFE')
    ),
    EBuncalibRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    algoRecover = cms.string('EcalRecHitWorkerRecover'),
    eeFEToBeRecovered = cms.InputTag("ecalDetIdToBeRecovered","eeFE"),
    EELaserMAX = cms.double(8.0),
    logWarningEtThreshold_EB_FE = cms.double(50),
    recoverEEIsolatedChannels = cms.bool(False)
)


process.ecalTestPulseUncalibRecHit = cms.EDProducer("EcalUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    algo = cms.string('EcalUncalibRecHitWorkerMaxSample'),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)


process.ecalCalibrationFilter = cms.EDFilter("EcalMonitorPrescaler",
    laserPrescaleFactor = cms.untracked.int32(1),
    testpulsePrescaleFactor = cms.untracked.int32(1),
    ledPrescaleFactor = cms.untracked.int32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    pedestalPrescaleFactor = cms.untracked.int32(1)
)


process.ecalLaserLedFilter = cms.EDFilter("EcalMonitorPrescaler",
    laserPrescaleFactor = cms.untracked.int32(1),
    ledPrescaleFactor = cms.untracked.int32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)


process.ecalPedestalFilter = cms.EDFilter("EcalMonitorPrescaler",
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    pedestalPrescaleFactor = cms.untracked.int32(1)
)


process.ecalTestPulseFilter = cms.EDFilter("EcalMonitorPrescaler",
    testpulsePrescaleFactor = cms.untracked.int32(1),
    EcalRawDataCollection = cms.InputTag("ecalDigis")
)


process.preScaler = cms.EDFilter("Prescaler",
    prescaleOffset = cms.int32(0),
    prescaleFactor = cms.int32(1)
)


process.dqmEnv = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('EcalCalibration'),
    eventRateWindow = cms.untracked.double(0.5),
    eventInfoFolder = cms.untracked.string('EventInfo')
)


process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('/home/dqmprolocal/output'),
    saveByTime = cms.untracked.int32(1),
    producer = cms.untracked.string('DQM'),
    saveByEvent = cms.untracked.int32(-1),
    forceRunNumber = cms.untracked.int32(-1),
    saveByRun = cms.untracked.int32(1),
    workflow = cms.untracked.string(''),
    saveAtJobEnd = cms.untracked.bool(False),
    fileFormat = cms.untracked.string('ROOT'),
    convention = cms.untracked.string('Online'),
    version = cms.untracked.int32(1),
    referenceRequireStatus = cms.untracked.int32(100),
    enableMultiThread = cms.untracked.bool(False),
    saveByMinute = cms.untracked.int32(8),
    filterName = cms.untracked.string(''),
    runIsComplete = cms.untracked.bool(False),
    saveByLumiSection = cms.untracked.int32(-1),
    referenceHandling = cms.untracked.string('all')
)


process.ecalCalibMonitorClient = cms.EDAnalyzer("EcalDQMonitorClient",
    moduleName = cms.untracked.string('EcalCalib Monitor Client'),
    PNMaskFile = cms.untracked.FileInPath('DQM/EcalMonitorClient/data/mask-PN.txt'),
    verbosity = cms.untracked.int32(0),
    workerParameters = cms.untracked.PSet(
        SummaryClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                TriggerPrimitives = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s emulator error quality summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of emulator matching quality. A tower is red if the number of events with Et emulation error is greater than 0.1 of total events. Towers with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                DesyncByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                ),
                IntegrityByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                ),
                Timing = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing quality summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the timing data quality. A 5x5 tower is red if the mean timing of the tower is off by more than 2.0 or RMS is greater than 6.0 (6.0 and 12.0 in forward region). Towers with total entries less than 15 are not subject to this evaluation. Since 5x5 tower timings are calculated with a tighter time-window than per-channel timings, a tower can additionally become red if its the sum of per-channel timing histogram entries is greater than per-tower histogram entries by factor 1.66666666667 (significant fraction of events fall outside the tight time-window).'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                HotCell = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the hot cell monitor. A channel is red if it has more than 100.0 times more entries than phi-ring mean in either digi, rec hit (filtered), or TP digi (filtered). Channels with less than 20 entries are not considered. Channel names of the hot cells are available in (Top) / Ecal / Errors / HotCells.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                RawData = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than 1.0 * log10(total entries).'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Presample = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Integrity = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                FEByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of front-ends in error status in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                )
            ),
            params = cms.untracked.PSet(
                activeSources = cms.untracked.vstring('Integrity', 
                    'RawData'),
                fedBadFraction = cms.untracked.double(0.5),
                towerBadFraction = cms.untracked.double(0.8)
            ),
            MEs = cms.untracked.PSet(
                ReportSummaryMap = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/EventInfo/reportSummaryMap'),
                    otype = cms.untracked.string('Ecal'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('DCC')
                ),
                ReportSummaryContents = cms.untracked.PSet(
                    kind = cms.untracked.string('REAL'),
                    description = cms.untracked.string(''),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Report'),
                    path = cms.untracked.string('Ecal/EventInfo/reportSummaryContents/Ecal_%(sm)s'),
                    perLumi = cms.untracked.bool(True)
                ),
                NBadFEDs = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Number of FEDs with more than 50.0% of channels in bad status. Updated every lumi section.'),
                    online = cms.untracked.bool(True),
                    otype = cms.untracked.string('None'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(1.0),
                        nbins = cms.untracked.int32(1),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('Ecal/Errors/Number of Bad Ecal FEDs')
                ),
                ReportSummary = cms.untracked.PSet(
                    kind = cms.untracked.string('REAL'),
                    description = cms.untracked.string(''),
                    otype = cms.untracked.string('Ecal'),
                    btype = cms.untracked.string('Report'),
                    path = cms.untracked.string('Ecal/EventInfo/reportSummary'),
                    perLumi = cms.untracked.bool(True)
                ),
                GlobalSummary = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Error summary used to trigger audio alarm. The value is identical to reportSummary.'),
                    online = cms.untracked.bool(True),
                    otype = cms.untracked.string('None'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(1.0),
                        nbins = cms.untracked.int32(1),
                        labels = cms.untracked.vstring('ECAL status'),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('Ecal/Errors/Global summary errors')
                ),
                QualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global summary%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the data quality. A channel is red if it is red in any one of RawData, Integrity, Timing, TriggerPrimitives, and HotCells task. A cluster of bad towers in this plot will cause the ReportSummary for the FED to go to 0 in online DQM.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        ),
        LedClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                Timing = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the mean led timing. Z scale is in LHC clocks. Due to the difference in pulse shape between led and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s')
                ),
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Mean led pulse amplitude in the PN diodes. In general, a PN channel is filled only when a led pulse was observed in the crystals that are associated to the diode. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('EESMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s')
                )
            ),
            params = cms.untracked.PSet(
                expectedTiming = cms.untracked.vdouble(4.4, 4.5),
                tolerancePNAmp = cms.untracked.double(0.1),
                toleranceTimRMS = cms.untracked.double(25.0),
                forwardFactor = cms.untracked.double(0.5),
                toleranceTiming = cms.untracked.double(1.0),
                expectedPNAmplitude = cms.untracked.vdouble(800.0, 800.0),
                tolerancePNRMSRatio = cms.untracked.double(1.0),
                toleranceAmpRMSRatio = cms.untracked.double(0.5),
                minChannelEntries = cms.untracked.int32(3),
                toleranceAmplitude = cms.untracked.double(0.1),
                expectedAmplitude = cms.untracked.vdouble(200.0, 10.0)
            ),
            MEs = cms.untracked.PSet(
                TimingMean = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Distribution of the timing in each crystal channel. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('EESM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(5.5),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(3.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led timing L%(wl)s %(sm)s')
                ),
                PNQualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Summary of the led data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0 for led 1 and 2 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EEMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT PN led quality summary L%(wl)s')
                ),
                TimingRMSMap = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the led timing RMS. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EE'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedClient/EELDT timing RMS L%(wl)s')
                ),
                AmplitudeMean = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('EESM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(400.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude L%(wl)s %(sm)s')
                ),
                QualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EE2P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT%(suffix)s led quality summary L%(wl)s')
                ),
                Quality = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led quality L%(wl)s %(sm)s')
                ),
                AmplitudeRMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EE'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude RMS L%(wl)s')
                )
            )
        ),
        LaserClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                Timing = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean laser timing. Z scale is in LHC clocks. Due to the difference in pulse shape between laser and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT timing %(sm)s L%(wl)s')
                ),
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Mean laser pulse amplitude in the PN diodes. In general, a PN channel is filled only when a laser pulse was observed in the crystals that are associated to the diode. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/PN/Gain16/%(prefix)sLT PNs amplitude %(sm)s G16 L%(wl)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude %(sm)s L%(wl)s')
                )
            ),
            params = cms.untracked.PSet(
                expectedTiming = cms.untracked.vdouble(4.2, 4.2, 4.2, 4.2),
                tolerancePNAmp = cms.untracked.double(0.1),
                toleranceTimRMS = cms.untracked.double(0.4),
                forwardFactor = cms.untracked.double(0.5),
                toleranceTiming = cms.untracked.double(0.5),
                expectedPNAmplitude = cms.untracked.vdouble(800.0, 800.0, 800.0, 800.0),
                tolerancePNRMSRatio = cms.untracked.double(1.0),
                toleranceAmpRMSRatio = cms.untracked.double(0.3),
                minChannelEntries = cms.untracked.int32(3),
                toleranceAmplitude = cms.untracked.double(0.1),
                expectedAmplitude = cms.untracked.vdouble(1700.0, 1300.0, 1700.0, 1700.0)
            ),
            MEs = cms.untracked.PSet(
                TimingRMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Distribution of the timing RMS in each crystal channel. X scale is in LHC clocks. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(0.5),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('rms (clock)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms L%(wl)s %(sm)s')
                ),
                TimingMean = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Distribution of the timing in each crystal channel. X scale is in LHC clocks. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(5.5),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(3.5),
                        title = cms.untracked.string('time (clock)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing mean L%(wl)s %(sm)s')
                ),
                PNQualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Summary of the laser data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0, 800.0, 800.0 for laser 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT PN laser quality summary L%(wl)s')
                ),
                TimingRMSMap = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the laser timing RMS. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal2P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('rms (clock)')
                    ),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms map L%(wl)s')
                ),
                AmplitudeMean = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(4096.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude L%(wl)s %(sm)s')
                ),
                QualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT%(suffix)s laser quality summary L%(wl)s')
                ),
                Quality = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser quality L%(wl)s %(sm)s')
                ),
                AmplitudeRMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal2P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('rms (ADC counts)')
                    ),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude rms L%(wl)s')
                )
            )
        ),
        PedestalClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                Pedestal = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('2D distribution of the mean pedestal.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s')
                ),
                PNPedestal = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Pedestal distribution of the PN diodes.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s')
                )
            ),
            params = cms.untracked.PSet(
                toleranceRMS = cms.untracked.vdouble(1.0, 1.2, 2.0),
                expectedMean = cms.untracked.double(200.0),
                tolerancePNRMS = cms.untracked.vdouble(20.0, 20.0),
                toleranceMean = cms.untracked.double(25.0),
                tolerancePNMean = cms.untracked.double(100.0),
                expectedPNMean = cms.untracked.double(750.0),
                minChannelEntries = cms.untracked.int32(3)
            ),
            MEs = cms.untracked.PSet(
                RMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Distribution of the pedestal RMS for each crystal channel. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal rms G%(gain)s %(sm)s')
                ),
                PNRMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Distribution of the pedestal RMS for each PN channel. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SMMEM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(50.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPDT PNs pedestal rms %(sm)s G%(pngain)s')
                ),
                PNQualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Summary of the pedestal quality for PN diodes. A channel is red if the pedestal mean is off from 750.0 by 100.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT PN pedestal quality G%(pngain)s summary')
                ),
                QualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT pedestal quality G%(gain)s summary%(suffix)s')
                ),
                Quality = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal quality G%(gain)s %(sm)s')
                ),
                Mean = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Distribution of pedestal mean in each channel. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(230.0),
                        nbins = cms.untracked.int32(120),
                        low = cms.untracked.double(170.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal mean G%(gain)s %(sm)s')
                )
            )
        ),
        RawDataClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                FEStatus = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('FE status counter.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.5),
                        nbins = cms.untracked.int32(16),
                        labels = cms.untracked.vstring('ENABLED', 
                            'DISABLED', 
                            'TIMEOUT', 
                            'HEADERERROR', 
                            'CHANNELID', 
                            'LINKERROR', 
                            'BLOCKSIZE', 
                            'SUPPRESSED', 
                            'FIFOFULL', 
                            'L1ADESYNC', 
                            'BXDESYNC', 
                            'L1ABXDESYNC', 
                            'FIFOFULLL1ADESYNC', 
                            'HPARITY', 
                            'VPARITY', 
                            'FORCEDZS'),
                        low = cms.untracked.double(-0.5)
                    ),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s')
                ),
                L1ADCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                Entries = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT DCC entries'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of entries recoreded by each FED'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                )
            ),
            params = cms.untracked.PSet(
                synchErrThresholdFactor = cms.untracked.double(1.0)
            ),
            MEs = cms.untracked.PSet(
                QualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than 1.0 * log10(total entries).'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                ErrorsSummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT front-end status errors summary'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Counter of data towers flagged as bad in the quality summary'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                )
            )
        ),
        PNIntegrityClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                MEMBlockSize = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemSize/%(prefix)sIT MemSize %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMChId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemChId/%(prefix)sIT MemChId %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMTowerId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemTTId/%(prefix)sIT MemTTId %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMGain = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemGain/%(prefix)sIT MemGain %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                )
            ),
            params = cms.untracked.PSet(
                errFractionThreshold = cms.untracked.double(0.01)
            ),
            MEs = cms.untracked.PSet(
                QualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary'),
                    otype = cms.untracked.string('MEM2P'),
                    description = cms.untracked.string('Summary of the data integrity in PN channels. A channel is red if more than 1.0% of its entries have integrity errors.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        ),
        IntegrityClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                BlockSize = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string('Digi occupancy.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Gain = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                GainSwitch = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                ChId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                TowerId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                )
            ),
            params = cms.untracked.PSet(
                errFractionThreshold = cms.untracked.double(0.01)
            ),
            MEs = cms.untracked.PSet(
                QualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Quality = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT data integrity quality %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        ),
        CalibrationSummaryClient = cms.untracked.PSet(
            sources = cms.untracked.PSet(
                Laser = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT%(suffix)s laser quality summary L%(wl)s')
                ),
                LaserPN = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Summary of the laser data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0, 800.0, 800.0 for laser 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT PN laser quality summary L%(wl)s')
                ),
                LedPN = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Summary of the led data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0 for led 1 and 2 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EEMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT PN led quality summary L%(wl)s')
                ),
                PedestalPN = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Summary of the pedestal quality for PN diodes. A channel is red if the pedestal mean is off from 750.0 by 100.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT PN pedestal quality G%(pngain)s summary')
                ),
                PNIntegrity = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary'),
                    otype = cms.untracked.string('MEM2P'),
                    description = cms.untracked.string('Summary of the data integrity in PN channels. A channel is red if more than 1.0% of its entries have integrity errors.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Led = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('EE2P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT%(suffix)s led quality summary L%(wl)s')
                ),
                Pedestal = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT pedestal quality G%(gain)s summary%(suffix)s')
                ),
                TestPulsePN = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Summary of test pulse data quality for PN diodes. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 12.5, 200.0 and 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary')
                ),
                TestPulse = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary')
                )
            ),
            params = cms.untracked.PSet(
                testPulseMGPAGains = cms.untracked.vint32(12),
                laserWavelengths = cms.untracked.vint32(1, 2, 3),
                pedestalMGPAGainsPN = cms.untracked.vint32(16),
                pedestalMGPAGains = cms.untracked.vint32(12),
                testPulseMGPAGainsPN = cms.untracked.vint32(16),
                ledWavelengths = cms.untracked.vint32(1, 2),
                activeSources = cms.untracked.vstring('Laser', 
                    'Led', 
                    'TestPulse')
            ),
            MEs = cms.untracked.PSet(
                QualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global calibration quality%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Summary of the calibration data quality. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                PNQualitySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s PN global quality'),
                    otype = cms.untracked.string('MEM2P'),
                    description = cms.untracked.string('Summary of the calibration data quality for the PN diodes. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        ),
        TestPulseClient = cms.untracked.PSet(
            toleranceRMS = cms.untracked.vdouble(160.0, 80.0, 10.0),
            amplitudeThreshold = cms.untracked.vdouble(1200.0, 600.0, 100.0),
            minChannelEntries = cms.untracked.int32(3),
            tolerancePNRMS = cms.untracked.vdouble(20.0, 20.0),
            sources = cms.untracked.PSet(
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Test pulse amplitude in the PN diodes.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Test pulse amplitude.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s')
                )
            ),
            MEs = cms.untracked.PSet(
                QualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary')
                ),
                PNQualitySummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Summary of test pulse data quality for PN diodes. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 12.5, 200.0 and 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('MEM2P'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary')
                ),
                Quality = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse quality G%(gain)s %(sm)s')
                ),
                AmplitudeRMS = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with entries less than 3 are not considered.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal2P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('rms (ADC counts)')
                    ),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse rms G%(gain)s')
                )
            ),
            PNAmplitudeThreshold = cms.untracked.vdouble(12.5, 200.0),
            name = cms.untracked.string('TestPulseClient')
        )
    ),
    workers = cms.untracked.vstring('IntegrityClient', 
        'RawDataClient', 
        'PedestalClient', 
        'TestPulseClient', 
        'LaserClient', 
        'LedClient', 
        'PNIntegrityClient', 
        'SummaryClient', 
        'CalibrationSummaryClient'),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalLaserLedMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string('EcalLaserLed Monitor Source'),
    resetInterval = cms.untracked.double(2.0),
    collectionTags = cms.untracked.PSet(
        TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
        EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
        TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
        EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
        PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
        EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
        EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
        Source = cms.untracked.InputTag("rawDataCollector"),
        MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
        MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
        EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
        TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
        EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
        EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
        EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
        EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
        EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
        EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
        EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EBSrFlag = cms.untracked.InputTag("ecalDigis"),
        BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
        EcalRawData = cms.untracked.InputTag("ecalDigis"),
        EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
        EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
        EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        EESrFlag = cms.untracked.InputTag("ecalDigis"),
        EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
    ),
    workers = cms.untracked.vstring('LaserTask', 
        'LedTask'),
    workerParameters = cms.untracked.PSet(
        LedTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                emptyLSLimit = cms.untracked.int32(3),
                ledWavelengths = cms.untracked.vint32(1, 2)
            ),
            MEs = cms.untracked.PSet(
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Mean led pulse amplitude in the PN diodes. In general, a PN channel is filled only when a led pulse was observed in the crystals that are associated to the diode. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('EESMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('EcalEndcap/EEOccupancyTask/EEOT led digi occupancy%(suffix)s'),
                    otype = cms.untracked.string('EE2P'),
                    description = cms.untracked.string('Led signal digi occupancy. Channels are filled whenever the DCC event type is set to LED.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Timing = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the mean led timing. Z scale is in LHC clocks. Due to the difference in pulse shape between led and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s')
                ),
                Shape = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('Led mean pulse shape. One slice corresponds to one readout tower (5x5 crystals). In general, a slice is filled only when a led pulse was observed in the tower. When no led signal was observed for longer than 3 lumi sections, the slices start to get filled with 0 amplitude, causing the shape to flatten.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(10),
                        low = cms.untracked.double(0.0)
                    ),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT shape %(sm)s L%(wl)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s')
                ),
                AOverP = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2)
                    ),
                    description = cms.untracked.string('2D distribution of the mean APD/PN value (event mean of per-event ratio).'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EESM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude over PN %(sm)s L%(wl)s')
                ),
                SignalRate = cms.untracked.PSet(
                    multi = cms.untracked.int32(2),
                    description = cms.untracked.string('Fraction of laser events with measurable led pulse.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('EE'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT signal rate L%(wl)s')
                ),
                AmplitudeSummary = cms.untracked.PSet(
                    multi = cms.untracked.int32(4),
                    description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('EE2P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude map L%(wl)s%(suffix)s')
                )
            )
        ),
        LaserTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                emptyLSLimit = cms.untracked.int32(3),
                laserWavelengths = cms.untracked.vint32(1, 2, 3)
            ),
            MEs = cms.untracked.PSet(
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Mean laser pulse amplitude in the PN diodes. In general, a PN channel is filled only when a laser pulse was observed in the crystals that are associated to the diode. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/PN/Gain16/%(prefix)sLT PNs amplitude %(sm)s G16 L%(wl)s')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT laser digi occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Laser signal digi occupancy. Channels are filled whenever the DCC event type is set to LASER.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Timing = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean laser timing. Z scale is in LHC clocks. Due to the difference in pulse shape between laser and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT timing %(sm)s L%(wl)s')
                ),
                Shape = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('Laser mean pulse shape. One slice corresponds to one readout tower (5x5 crystals). In general, a slice is filled only when a laser pulse was observed in the tower. When no laser signal was observed for longer than 3 lumi sections, the slices start to get filled with 0 amplitude, causing the shape to flatten.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(10),
                        low = cms.untracked.double(0.0)
                    ),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT shape %(sm)s L%(wl)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude %(sm)s L%(wl)s')
                ),
                AOverP = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean APD/PN value (event mean of per-event ratio).'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude over PN %(sm)s L%(wl)s')
                ),
                SignalRate = cms.untracked.PSet(
                    multi = cms.untracked.int32(4),
                    description = cms.untracked.string('Fraction of laser events with measurable laser pulse.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT signal rate L%(wl)s')
                ),
                AmplitudeSummary = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        wl = cms.untracked.vint32(1, 2, 3)
                    ),
                    description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude map L%(wl)s%(suffix)s')
                )
            )
        )
    ),
    verbosity = cms.untracked.int32(0),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalMEFormatter = cms.EDAnalyzer("EcalMEFormatter",
    verbosity = cms.untracked.int32(0),
    MEs = cms.untracked.PSet(
        ClusterTaskBCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection phi%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        ClusterTaskTrendNBC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of basic clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the number of basic clusters per event in EB/EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        ClusterTaskBCSizeMapProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection eta%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('ProjEta')
        ),
        ClusterTaskBCSize = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the basic cluster size (number of crystals).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size')
        ),
        ClusterTaskBCEtMapProjEta = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection eta%(suffix)s')
        ),
        ClusterTaskSCR9 = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of E_seed / E_3x3 of the super clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.2),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC R9')
        ),
        ClusterTaskTrendSCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of super clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the mean size (number of crystals) of the super clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        ClusterTaskTrendNSC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of super clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the number of super clusters per event in EB/EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        ClusterTaskSCNum = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of super clusters per event in EB/EE.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC number')
        ),
        ClusterTaskSCSeedOccupancyHighE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (high energy clusters) %(supercrystal)s binned'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters with energy > 2.0 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ClusterTaskSCE = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Super cluster energy distribution.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy')
        ),
        ClusterTaskSCSizeVsEnergy = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Mean SC size in crystals as a function of the SC energy.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC size (crystal) vs energy (GeV)')
        ),
        ClusterTaskSCSeedOccupancyTrig = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (%(trig)s triggered) %(supercrystal)s binned')
        ),
        ClusterTaskSCSeedEnergy = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Energy distribution of the crystals that seeded super clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed crystal energy')
        ),
        ClusterTaskSCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_eta'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Supercluster eta.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        ClusterTaskSCSeedTimeMapTrigEx = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Mean timing of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing map%(suffix)s (%(trig)s exclusive triggered) %(supercrystal)s binned')
        ),
        ClusterTaskBCE = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Basic cluster energy distribution.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy')
        ),
        ClusterTaskBCSizeMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('2D distribution of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ClusterTaskSCNcrystals = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the super cluster size (number of crystals).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size (crystal)')
        ),
        ClusterTaskBCNum = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of basic clusters per event.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number')
        ),
        ClusterTaskSCSeedTimeTrigEx = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Timing distribution of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing (%(trig)s exclusive triggered)')
        ),
        ClusterTaskSCSwissCross = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Swiss cross for SC maximum-energy crystal.'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBRecoSummary/superClusters_EB_E1oE4')
        ),
        ClusterTaskSingleCrystalCluster = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC single crystal cluster seed occupancy map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the occurrence of super clusters with only one constituent'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ClusterTaskTriggers = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Counter for the trigger categories'),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(5),
                labels = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('triggers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE triggers')
        ),
        ClusterTaskBCSizeMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection phi%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('ProjPhi')
        ),
        ClusterTaskBCOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Basic cluster occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ClusterTaskBCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection eta%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        ClusterTaskSCNBCs = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the super cluster size (number of basic clusters)'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.0),
                nbins = cms.untracked.int32(15),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size')
        ),
        ClusterTaskBCEtMapProjPhi = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection phi%(suffix)s')
        ),
        ClusterTaskSCSeedOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed occupancy map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ClusterTaskSCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_phi'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Supercluster phi.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        ClusterTaskBCEMapProjEta = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection eta%(suffix)s')
        ),
        ClusterTaskSCELow = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Energy distribution of the super clusters (low scale).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy (low scale)')
        ),
        ClusterTaskTrendBCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of basic clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the mean size of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        ClusterTaskExclusiveTriggers = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Counter for the trigger categories'),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(5),
                labels = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('triggers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE exclusive triggers')
        ),
        ClusterTaskBCEMapProjPhi = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection phi%(suffix)s')
        ),
        ClusterTaskSCClusterVsSeed = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Relation between super cluster energy and its seed crystal energy.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy vs seed crystal energy')
        ),
        ClusterTaskBCEMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean energy of the basic clusters.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy map%(suffix)s')
        ),
        EnergyTaskHitMapAll = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s energy summary')
        ),
        EnergyTaskHitAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rec hit energy distribution.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit spectrum%(suffix)s')
        ),
        EnergyTaskHit = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rec hit energy distribution.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT energy spectrum %(sm)s')
        ),
        EnergyTaskHitMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit energy %(sm)s')
        ),
        IntegrityTaskTotal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Total number of integrity errors for each FED.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        IntegrityTaskBlockSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        IntegrityTaskByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        IntegrityTaskGain = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        IntegrityTaskGainSwitch = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        IntegrityTaskTrendNErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/IntegrityTask number of integrity errors'),
            otype = cms.untracked.string('Ecal'),
            description = cms.untracked.string('Trend of the number of integrity errors.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('Trend')
        ),
        IntegrityTaskChId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        IntegrityTaskTowerId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        OccupancyTaskTrendNTPDigi = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered TP digis'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        OccupancyTaskRecHitThr1D = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(500.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of filtered rec hits in event')
        ),
        OccupancyTaskDigi1D = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of digis per event.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3000.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of digis in event')
        ),
        OccupancyTaskDigiAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        OccupancyTaskTPDigiThrProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        OccupancyTaskRecHitThrProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        OccupancyTaskRecHitProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of all rec hits.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        OccupancyTaskDigiProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        OccupancyTaskTrendNRecHitThr = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered recHits'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        OccupancyTaskTPDigiThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        OccupancyTaskDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT DCC entries'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of entries recoreded by each FED'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        OccupancyTaskTPDigiThrProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        OccupancyTaskTrendNDigi = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of digis'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of digis.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        OccupancyTaskDigi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        OccupancyTaskDigiDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT digi occupancy summary 1D'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('DCC digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        OccupancyTaskRecHitAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Rec hit occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        OccupancyTaskRecHitProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the rec hit occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        OccupancyTaskRecHitThrProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        OccupancyTaskDigiProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        OccupancyTaskRecHitThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        PresampleTaskPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('2D distribution of mean presample value.'),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal')
        ),
        RawDataTaskBXSRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing SRP errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and SRP.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskCRC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT CRC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of CRC errors.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskBXFE = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE errors')
        ),
        RawDataTaskBXDCCDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC-GT')
        ),
        RawDataTaskBXFEDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE-DCC')
        ),
        RawDataTaskOrbitDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number DCC-GT')
        ),
        RawDataTaskL1ASRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and SRP.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskBXTCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing TCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of bunch corssing value mismatches between DCC and TCC.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskDesyncTotal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT total FE synchronization errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskRunNumber = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT run number errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between run numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskOrbit = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskBXDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskBXFEInvalid = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(69.0),
                nbins = cms.untracked.int32(69),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing invalid value')
        ),
        RawDataTaskDesyncByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        RawDataTaskL1ATCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A TCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and TCC.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskFEByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of front-ends in error status in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        RawDataTaskTrendNSyncErrors = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Accumulated trend of the number of synchronization errors (L1A & BX mismatches) between DCC and FE in this run.'),
            cumulative = cms.untracked.bool(True),
            btype = cms.untracked.string('Trend'),
            otype = cms.untracked.string('Ecal'),
            online = cms.untracked.bool(True),
            path = cms.untracked.string('Ecal/Trends/RawDataTask accumulated number of sync errors')
        ),
        RawDataTaskEventTypePostCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing > 3490.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX')
        ),
        RawDataTaskL1ADCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskEventTypePreCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing < 3490'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type pre calibration BX')
        ),
        RawDataTaskEventTypeCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing == 3490. This plot is filled using data from the physics data stream during physics runs. It is normal to have very few entries in these cases.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX')
        ),
        RawDataTaskL1AFE = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A FE errors')
        ),
        RawDataTaskTriggerType = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT trigger type errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between trigger type recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RawDataTaskFEStatus = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('FE status counter.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('ENABLED', 
                    'DISABLED', 
                    'TIMEOUT', 
                    'HEADERERROR', 
                    'CHANNELID', 
                    'LINKERROR', 
                    'BLOCKSIZE', 
                    'SUPPRESSED', 
                    'FIFOFULL', 
                    'L1ADESYNC', 
                    'BXDESYNC', 
                    'L1ABXDESYNC', 
                    'FIFOFULLL1ADESYNC', 
                    'HPARITY', 
                    'VPARITY', 
                    'FORCEDZS'),
                low = cms.untracked.double(-0.5)
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s')
        ),
        RecoSummaryTaskRecoFlagReduced = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from reduced rechits.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/redRecHits_%(subdetshort)s_recoFlag')
        ),
        RecoSummaryTaskChi2 = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Chi2 of the pulse reconstruction.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_Chi2')
        ),
        RecoSummaryTaskRecoFlagBasicCluster = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from rechits in basic clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/basicClusters_recHits_%(subdetshort)s_recoFlag')
        ),
        RecoSummaryTaskSwissCross = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Swiss cross.'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_E1oE4')
        ),
        RecoSummaryTaskRecoFlagAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from all rechits.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_recoFlag')
        ),
        RecoSummaryTaskTime = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rechit time.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-50.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_time')
        ),
        RecoSummaryTaskEnergyMax = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Maximum energy of the rechit.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(110),
                low = cms.untracked.double(-10.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_energyMax')
        ),
        TrigPrimTaskLowIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of low interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        TrigPrimTaskHighIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of high interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        TrigPrimTaskEtReal = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et spectrum Real Digis%(suffix)s')
        ),
        TrigPrimTaskEtVsBx = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Mean TP Et in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP Et')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('1', 
                    '271', 
                    '541', 
                    '892', 
                    '1162', 
                    '1432', 
                    '1783', 
                    '2053', 
                    '2323', 
                    '2674', 
                    '2944', 
                    '3214', 
                    '3446', 
                    '3490', 
                    '3491', 
                    '3565'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s')
        ),
        TrigPrimTaskMedIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of medium interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        TrigPrimTaskTTFlags = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Distribution of the trigger tower flags.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(7.5),
                nbins = cms.untracked.int32(8),
                labels = cms.untracked.vstring('0', 
                    '1', 
                    '2', 
                    '3', 
                    '4', 
                    '5', 
                    '6', 
                    '7'),
                low = cms.untracked.double(-0.5),
                title = cms.untracked.string('TT flag')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT Flags%(suffix)s')
        ),
        TrigPrimTaskTTFMismatch = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT flag mismatch%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        TrigPrimTaskEtSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Et trigger tower summary')
        ),
        TrigPrimTaskEtRealMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et map Real Digis %(sm)s')
        ),
        TrigPrimTaskOccVsBx = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('TP occupancy in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('1', 
                    '271', 
                    '541', 
                    '892', 
                    '1162', 
                    '1432', 
                    '1783', 
                    '2053', 
                    '2323', 
                    '2674', 
                    '2944', 
                    '3214', 
                    '3446', 
                    '3490', 
                    '3491', 
                    '3565'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TP occupancy vs bx Real Digis%(suffix)s')
        ),
        IntegrityClientQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        IntegrityClientQuality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT data integrity quality %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        OccupancyClientQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the hot cell monitor. A channel is red if it has more than 100.0 times more entries than phi-ring mean in either digi, rec hit (filtered), or TP digi (filtered). Channels with less than 20 entries are not considered. Channel names of the hot cells are available in (Top) / Ecal / Errors / HotCells.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        PresampleClientRMS = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the presample RMS of each channel. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms G12 %(sm)s')
        ),
        PresampleClientTrendRMS = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal rms max'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of presample RMS averaged over all channels in EB / EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        PresampleClientRMSMap = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms map G12 %(sm)s')
        ),
        PresampleClientTrendMean = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal mean max - min'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of presample spread in EB / EE. Y value indicates the difference between maximum and minimum presample mean values within the subdetector.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        PresampleClientRMSMapAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 RMS map')
        ),
        PresampleClientQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        PresampleClientQuality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal quality G12 %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        PresampleClientErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT pedestal quality errors summary G12'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Counter of channels flagged as bad in the quality summary'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        PresampleClientMean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('1D distribution of the mean presample value in each crystal. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(230.0),
                nbins = cms.untracked.int32(120),
                low = cms.untracked.double(170.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal mean G12 %(sm)s')
        ),
        RawDataClientQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than 1.0 * log10(total entries).'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        RawDataClientErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT front-end status errors summary'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Counter of data towers flagged as bad in the quality summary'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        SummaryClientReportSummaryMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryMap'),
            otype = cms.untracked.string('Ecal'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('DCC')
        ),
        SummaryClientReportSummaryContents = cms.untracked.PSet(
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string(''),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryContents/Ecal_%(sm)s'),
            perLumi = cms.untracked.bool(True)
        ),
        SummaryClientNBadFEDs = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Number of FEDs with more than 50.0% of channels in bad status. Updated every lumi section.'),
            online = cms.untracked.bool(True),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0),
                nbins = cms.untracked.int32(1),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('Ecal/Errors/Number of Bad Ecal FEDs')
        ),
        SummaryClientReportSummary = cms.untracked.PSet(
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string(''),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            path = cms.untracked.string('Ecal/EventInfo/reportSummary'),
            perLumi = cms.untracked.bool(True)
        ),
        SummaryClientGlobalSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Error summary used to trigger audio alarm. The value is identical to reportSummary.'),
            online = cms.untracked.bool(True),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0),
                nbins = cms.untracked.int32(1),
                labels = cms.untracked.vstring('ECAL status'),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('Ecal/Errors/Global summary errors')
        ),
        SummaryClientQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global summary%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the data quality. A channel is red if it is red in any one of RawData, Integrity, Timing, TriggerPrimitives, and HotCells task. A cluster of bad towers in this plot will cause the ReportSummary for the FED to go to 0 in online DQM.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    )
)


process.ecalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string('Ecal Monitor Source'),
    resetInterval = cms.untracked.double(2.0),
    collectionTags = cms.untracked.PSet(
        TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
        EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
        TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
        EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
        PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
        EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
        EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
        Source = cms.untracked.InputTag("rawDataCollector"),
        MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
        MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
        EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
        TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
        EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
        EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
        EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
        EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
        EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
        EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
        EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EBSrFlag = cms.untracked.InputTag("ecalDigis"),
        BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
        EcalRawData = cms.untracked.InputTag("ecalDigis"),
        EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
        EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
        EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        EESrFlag = cms.untracked.InputTag("ecalDigis"),
        EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
    ),
    verbosity = cms.untracked.int32(0),
    workers = cms.untracked.vstring('IntegrityTask', 
        'RawDataTask'),
    evaluateTime = cms.untracked.bool(False),
    allowMissingCollections = cms.untracked.bool(True),
    workerParameters = cms.untracked.PSet(
        EnergyTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                isPhysicsRun = cms.untracked.bool(True)
            ),
            MEs = cms.untracked.PSet(
                HitMapAll = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s energy summary')
                ),
                HitAll = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Rec hit energy distribution.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(20.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit spectrum%(suffix)s')
                ),
                Hit = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Rec hit energy distribution.'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(20.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT energy spectrum %(sm)s')
                ),
                HitMap = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
                    otype = cms.untracked.string('SM'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit energy %(sm)s')
                )
            )
        ),
        RecoSummaryTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                rechitThresholdEE = cms.untracked.double(1.2),
                rechitThresholdEB = cms.untracked.double(0.8)
            ),
            MEs = cms.untracked.PSet(
                RecoFlagReduced = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Reconstruction flags from reduced rechits.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.5),
                        nbins = cms.untracked.int32(16),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/redRecHits_%(subdetshort)s_recoFlag')
                ),
                Chi2 = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Chi2 of the pulse reconstruction.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_Chi2')
                ),
                RecoFlagBasicCluster = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Reconstruction flags from rechits in basic clusters.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.5),
                        nbins = cms.untracked.int32(16),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/basicClusters_recHits_%(subdetshort)s_recoFlag')
                ),
                SwissCross = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Swiss cross.'),
                    otype = cms.untracked.string('EB'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(1.5),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_E1oE4')
                ),
                RecoFlagAll = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Reconstruction flags from all rechits.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.5),
                        nbins = cms.untracked.int32(16),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_recoFlag')
                ),
                Time = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Rechit time.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(50.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-50.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_time')
                ),
                EnergyMax = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Maximum energy of the rechit.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(110),
                        low = cms.untracked.double(-10.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_energyMax')
                )
            )
        ),
        PresampleTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                nSamples = cms.untracked.int32(3),
                pulseMaxPosition = cms.untracked.int32(5)
            ),
            MEs = cms.untracked.PSet(
                Pedestal = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string('2D distribution of mean presample value.'),
                    kind = cms.untracked.string('TProfile2D'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        ),
        ClusterTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                swissCrossMaxThreshold = cms.untracked.double(3.0),
                L1MuGMTReadoutCollectionTag = cms.untracked.InputTag("gtDigis"),
                egTriggerAlgos = cms.untracked.vstring('L1_SingleEG2', 
                    'L1_SingleEG5', 
                    'L1_SingleEG8', 
                    'L1_SingleEG10', 
                    'L1_SingleEG12', 
                    'L1_SingleEG15', 
                    'L1_SingleEG20', 
                    'L1_SingleEG25', 
                    'L1_DoubleNoIsoEG_BTB_tight', 
                    'L1_DoubleNoIsoEG_BTB_loose', 
                    'L1_DoubleNoIsoEGTopBottom', 
                    'L1_DoubleNoIsoEGTopBottomCen', 
                    'L1_DoubleNoIsoEGTopBottomCen2', 
                    'L1_DoubleNoIsoEGTopBottomCenVert'),
                L1GlobalTriggerReadoutRecordTag = cms.untracked.InputTag("gtDigis"),
                doExtra = cms.untracked.bool(True),
                energyThreshold = cms.untracked.double(2.0)
            ),
            MEs = cms.untracked.PSet(
                BCOccupancyProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection phi%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the basic cluster occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                TrendNBC = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of basic clusters'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the number of basic clusters per event in EB/EE.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                BCSizeMapProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection eta%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('ProjEta')
                ),
                BCSize = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the basic cluster size (number of crystals).'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size')
                ),
                BCEtMapProjEta = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('transverse energy (GeV)')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('ProjEta'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection eta%(suffix)s')
                ),
                SCR9 = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of E_seed / E_3x3 of the super clusters.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(1.2),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC R9')
                ),
                TrendSCSize = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of super clusters'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the mean size (number of crystals) of the super clusters.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                TrendNSC = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of super clusters'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the number of super clusters per event in EB/EE.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                SCNum = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the number of super clusters per event in EB/EE.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(20.0),
                        nbins = cms.untracked.int32(20),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC number')
                ),
                SCSeedOccupancyHighE = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (high energy clusters) %(supercrystal)s binned'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters with energy > 2.0 GeV.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                SCE = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Super cluster energy distribution.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy')
                ),
                SCSizeVsEnergy = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Mean SC size in crystals as a function of the SC energy.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC size (crystal) vs energy (GeV)')
                ),
                SCSeedOccupancyTrig = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        trig = cms.untracked.vstring('ECAL', 
                            'HCAL', 
                            'CSC', 
                            'DT', 
                            'RPC')
                    ),
                    description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
                    kind = cms.untracked.string('TH2F'),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (%(trig)s triggered) %(supercrystal)s binned')
                ),
                SCSeedEnergy = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Energy distribution of the crystals that seeded super clusters.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed crystal energy')
                ),
                SCOccupancyProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_eta'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Supercluster eta.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                SCSeedTimeMapTrigEx = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        trig = cms.untracked.vstring('ECAL', 
                            'HCAL', 
                            'CSC', 
                            'DT', 
                            'RPC')
                    ),
                    description = cms.untracked.string('Mean timing of the crystals that seeded super clusters.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        high = cms.untracked.double(25.0),
                        low = cms.untracked.double(-25.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing map%(suffix)s (%(trig)s exclusive triggered) %(supercrystal)s binned')
                ),
                BCE = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Basic cluster energy distribution.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy')
                ),
                BCSizeMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size map%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('2D distribution of the mean size (number of crystals) of the basic clusters.'),
                    kind = cms.untracked.string('TProfile2D'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                SCNcrystals = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the super cluster size (number of crystals).'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size (crystal)')
                ),
                BCNum = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the number of basic clusters per event.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(20),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number')
                ),
                SCSeedTimeTrigEx = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        trig = cms.untracked.vstring('ECAL', 
                            'HCAL', 
                            'CSC', 
                            'DT', 
                            'RPC')
                    ),
                    description = cms.untracked.string('Timing distribution of the crystals that seeded super clusters.'),
                    kind = cms.untracked.string('TH1F'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(25.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-25.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing (%(trig)s exclusive triggered)')
                ),
                SCSwissCross = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Swiss cross for SC maximum-energy crystal.'),
                    otype = cms.untracked.string('EB'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(1.5),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('EcalBarrel/EBRecoSummary/superClusters_EB_E1oE4')
                ),
                SingleCrystalCluster = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC single crystal cluster seed occupancy map%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy map of the occurrence of super clusters with only one constituent'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Triggers = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Counter for the trigger categories'),
                    otype = cms.untracked.string('None'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(5.0),
                        nbins = cms.untracked.int32(5),
                        labels = cms.untracked.vstring('ECAL', 
                            'HCAL', 
                            'CSC', 
                            'DT', 
                            'RPC'),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('triggers')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE triggers')
                ),
                BCSizeMapProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection phi%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                BCOccupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number map%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Basic cluster occupancy.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                BCOccupancyProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection eta%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the basic cluster occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                SCNBCs = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the super cluster size (number of basic clusters)'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.0),
                        nbins = cms.untracked.int32(15),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size')
                ),
                BCEtMapProjPhi = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('transverse energy (GeV)')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('ProjPhi'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection phi%(suffix)s')
                ),
                SCSeedOccupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed occupancy map%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                SCOccupancyProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_phi'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Supercluster phi.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                BCEMapProjEta = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('ProjEta'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection eta%(suffix)s')
                ),
                SCELow = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Energy distribution of the super clusters (low scale).'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy (low scale)')
                ),
                TrendBCSize = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of basic clusters'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the mean size of the basic clusters.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                ExclusiveTriggers = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Counter for the trigger categories'),
                    otype = cms.untracked.string('None'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(5.0),
                        nbins = cms.untracked.int32(5),
                        labels = cms.untracked.vstring('ECAL', 
                            'HCAL', 
                            'CSC', 
                            'DT', 
                            'RPC'),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('triggers')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE exclusive triggers')
                ),
                BCEMapProjPhi = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('ProjPhi'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection phi%(suffix)s')
                ),
                SCClusterVsSeed = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Relation between super cluster energy and its seed crystal energy.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(150.0),
                        nbins = cms.untracked.int32(50),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy vs seed crystal energy')
                ),
                BCEMap = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean energy of the basic clusters.'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy map%(suffix)s')
                )
            )
        ),
        RawDataTask = cms.untracked.PSet(
            MEs = cms.untracked.PSet(
                BXSRP = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing SRP errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and SRP.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                CRC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT CRC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of CRC errors.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                BXFE = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(68.0),
                        nbins = cms.untracked.int32(68),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('iFE')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE errors')
                ),
                BXDCCDiff = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(200),
                        low = cms.untracked.double(-100.0)
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC-GT')
                ),
                BXFEDiff = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(200),
                        low = cms.untracked.double(-100.0)
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE-DCC')
                ),
                OrbitDiff = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(100.0),
                        nbins = cms.untracked.int32(200),
                        low = cms.untracked.double(-100.0)
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number DCC-GT')
                ),
                L1ASRP = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of L1A value mismatches between DCC and SRP.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                BXTCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing TCC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of bunch corssing value mismatches between DCC and TCC.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                DesyncTotal = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT total FE synchronization errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                RunNumber = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT run number errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between run numbers recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                Orbit = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                BXDCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                BXFEInvalid = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(69.0),
                        nbins = cms.untracked.int32(69),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('iFE')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing invalid value')
                ),
                DesyncByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                ),
                L1ATCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A TCC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of L1A value mismatches between DCC and TCC.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                FEByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of front-ends in error status in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                ),
                TrendNSyncErrors = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Accumulated trend of the number of synchronization errors (L1A & BX mismatches) between DCC and FE in this run.'),
                    cumulative = cms.untracked.bool(True),
                    btype = cms.untracked.string('Trend'),
                    otype = cms.untracked.string('Ecal'),
                    online = cms.untracked.bool(True),
                    path = cms.untracked.string('Ecal/Trends/RawDataTask accumulated number of sync errors')
                ),
                EventTypePostCalib = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing > 3490.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(24.5),
                        nbins = cms.untracked.int32(25),
                        labels = cms.untracked.vstring('UNKNOWN', 
                            'COSMIC', 
                            'BEAMH4', 
                            'BEAMH2', 
                            'MTCC', 
                            'LASER_STD', 
                            'LASER_POWER_SCAN', 
                            'LASER_DELAY_SCAN', 
                            'TESTPULSE_SCAN_MEM', 
                            'TESTPULSE_MGPA', 
                            'PEDESTAL_STD', 
                            'PEDESTAL_OFFSET_SCAN', 
                            'PEDESTAL_25NS_SCAN', 
                            'LED_STD', 
                            'PHYSICS_GLOBAL', 
                            'COSMICS_GLOBAL', 
                            'HALO_GLOBAL', 
                            'LASER_GAP', 
                            'TESTPULSE_GAP', 
                            'PEDESTAL_GAP', 
                            'LED_GAP', 
                            'PHYSICS_LOCAL', 
                            'COSMICS_LOCAL', 
                            'HALO_LOCAL', 
                            'CALIB_LOCAL'),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX')
                ),
                L1ADCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                EventTypePreCalib = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing < 3490'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(24.5),
                        nbins = cms.untracked.int32(25),
                        labels = cms.untracked.vstring('UNKNOWN', 
                            'COSMIC', 
                            'BEAMH4', 
                            'BEAMH2', 
                            'MTCC', 
                            'LASER_STD', 
                            'LASER_POWER_SCAN', 
                            'LASER_DELAY_SCAN', 
                            'TESTPULSE_SCAN_MEM', 
                            'TESTPULSE_MGPA', 
                            'PEDESTAL_STD', 
                            'PEDESTAL_OFFSET_SCAN', 
                            'PEDESTAL_25NS_SCAN', 
                            'LED_STD', 
                            'PHYSICS_GLOBAL', 
                            'COSMICS_GLOBAL', 
                            'HALO_GLOBAL', 
                            'LASER_GAP', 
                            'TESTPULSE_GAP', 
                            'PEDESTAL_GAP', 
                            'LED_GAP', 
                            'PHYSICS_LOCAL', 
                            'COSMICS_LOCAL', 
                            'HALO_LOCAL', 
                            'CALIB_LOCAL'),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type pre calibration BX')
                ),
                EventTypeCalib = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing == 3490. This plot is filled using data from the physics data stream during physics runs. It is normal to have very few entries in these cases.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(24.5),
                        nbins = cms.untracked.int32(25),
                        labels = cms.untracked.vstring('UNKNOWN', 
                            'COSMIC', 
                            'BEAMH4', 
                            'BEAMH2', 
                            'MTCC', 
                            'LASER_STD', 
                            'LASER_POWER_SCAN', 
                            'LASER_DELAY_SCAN', 
                            'TESTPULSE_SCAN_MEM', 
                            'TESTPULSE_MGPA', 
                            'PEDESTAL_STD', 
                            'PEDESTAL_OFFSET_SCAN', 
                            'PEDESTAL_25NS_SCAN', 
                            'LED_STD', 
                            'PHYSICS_GLOBAL', 
                            'COSMICS_GLOBAL', 
                            'HALO_GLOBAL', 
                            'LASER_GAP', 
                            'TESTPULSE_GAP', 
                            'PEDESTAL_GAP', 
                            'LED_GAP', 
                            'PHYSICS_LOCAL', 
                            'COSMICS_LOCAL', 
                            'HALO_LOCAL', 
                            'CALIB_LOCAL'),
                        low = cms.untracked.double(-0.5)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX')
                ),
                L1AFE = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Number of L1A value mismatches between DCC and FE.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(68.0),
                        nbins = cms.untracked.int32(68),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('iFE')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A FE errors')
                ),
                TriggerType = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT trigger type errors'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of discrepancies between trigger type recorded in the DCC and that in CMS Event.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                FEStatus = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('FE status counter.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(15.5),
                        nbins = cms.untracked.int32(16),
                        labels = cms.untracked.vstring('ENABLED', 
                            'DISABLED', 
                            'TIMEOUT', 
                            'HEADERERROR', 
                            'CHANNELID', 
                            'LINKERROR', 
                            'BLOCKSIZE', 
                            'SUPPRESSED', 
                            'FIFOFULL', 
                            'L1ADESYNC', 
                            'BXDESYNC', 
                            'L1ABXDESYNC', 
                            'FIFOFULLL1ADESYNC', 
                            'HPARITY', 
                            'VPARITY', 
                            'FORCEDZS'),
                        low = cms.untracked.double(-0.5)
                    ),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s')
                )
            )
        ),
        TimingTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                energyThresholdEE = cms.untracked.double(3.0),
                energyThresholdEB = cms.untracked.double(1.0)
            ),
            MEs = cms.untracked.PSet(
                TimeMap = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 25.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
                    otype = cms.untracked.string('SM'),
                    zaxis = cms.untracked.PSet(
                        high = cms.untracked.double(25.0),
                        low = cms.untracked.double(-25.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(sm)s')
                ),
                TimeAll = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(25.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-25.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D summary%(suffix)s')
                ),
                TimeAllMap = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 7.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        high = cms.untracked.double(7.0),
                        low = cms.untracked.double(-7.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing map%(suffix)s')
                ),
                TimeAmpAll = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(50.0),
                        nbins = cms.untracked.int32(200),
                        low = cms.untracked.double(-50.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        edges = cms.untracked.vdouble(0.316227766017, 0.354813389234, 0.398107170553, 0.446683592151, 0.501187233627, 
                            0.56234132519, 0.63095734448, 0.707945784384, 0.794328234724, 0.891250938134, 
                            1.0, 1.1220184543, 1.25892541179, 1.41253754462, 1.58489319246, 
                            1.77827941004, 1.99526231497, 2.23872113857, 2.51188643151, 2.81838293126, 
                            3.16227766017, 3.54813389234, 3.98107170553, 4.46683592151, 5.01187233627, 
                            5.6234132519, 6.3095734448, 7.07945784384, 7.94328234724, 8.91250938134, 
                            10.0, 11.220184543, 12.5892541179, 14.1253754462, 15.8489319246, 
                            17.7827941004, 19.9526231497, 22.3872113857, 25.1188643151, 28.1838293126, 
                            31.6227766017, 35.4813389234, 39.8107170553, 44.6683592151, 50.1187233627, 
                            56.234132519, 63.095734448, 70.7945784384, 79.4328234724, 89.1250938134),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude summary%(suffix)s')
                ),
                TimeAmp = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(50.0),
                        nbins = cms.untracked.int32(200),
                        low = cms.untracked.double(-50.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        edges = cms.untracked.vdouble(0.316227766017, 0.354813389234, 0.398107170553, 0.446683592151, 0.501187233627, 
                            0.56234132519, 0.63095734448, 0.707945784384, 0.794328234724, 0.891250938134, 
                            1.0, 1.1220184543, 1.25892541179, 1.41253754462, 1.58489319246, 
                            1.77827941004, 1.99526231497, 2.23872113857, 2.51188643151, 2.81838293126, 
                            3.16227766017, 3.54813389234, 3.98107170553, 4.46683592151, 5.01187233627, 
                            5.6234132519, 6.3095734448, 7.07945784384, 7.94328234724, 8.91250938134, 
                            10.0, 11.220184543, 12.5892541179, 14.1253754462, 15.8489319246, 
                            17.7827941004, 19.9526231497, 22.3872113857, 25.1188643151, 28.1838293126, 
                            31.6227766017, 35.4813389234, 39.8107170553, 44.6683592151, 50.1187233627, 
                            56.234132519, 63.095734448, 70.7945784384, 79.4328234724, 89.1250938134),
                        title = cms.untracked.string('energy (GeV)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude %(sm)s')
                ),
                Time1D = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
                    otype = cms.untracked.string('SM'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(25.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-25.0),
                        title = cms.untracked.string('time (ns)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D %(sm)s')
                )
            )
        ),
        SelectiveReadoutTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                DCCZS1stSample = cms.untracked.int32(2),
                useCondDb = cms.untracked.bool(False),
                ZSFIRWeights = cms.untracked.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 
                    0.3707)
            ),
            MEs = cms.untracked.PSet(
                HighIntOutput = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Output of the ZS filter for high interest towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(60.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-60.0),
                        title = cms.untracked.string('ADC counts*4')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest ZS filter output%(suffix)s')
                ),
                ZS1Map = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1 counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy with ZS1 flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                FRDropped = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Number of FR flagged but dropped towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(20.0),
                        nbins = cms.untracked.int32(20),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('number of towers')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout Number%(suffix)s')
                ),
                ZSFullReadout = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Number of ZS flagged but fully read out towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(20.0),
                        nbins = cms.untracked.int32(20),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('number of towers')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout Number%(suffix)s')
                ),
                ZSFullReadoutMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT ZS flagged full readout counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Number of ZS flagged but fully read out towers.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                FRDroppedMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT FR flagged dropped counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Number of FR flagged but dropped towers.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                LowIntOutput = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Output of the ZS filter for low interest towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(60.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(-60.0),
                        title = cms.untracked.string('ADC counts*4')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest ZS filter output%(suffix)s')
                ),
                LowIntPayload = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total data size from all low interest towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(3.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('event size (kB)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest payload%(suffix)s')
                ),
                TowerSize = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the mean data size from each readout unit.'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        title = cms.untracked.string('size (bytes)')
                    ),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT tower event size%(suffix)s')
                ),
                DCCSize = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Distribution of the per-DCC data size.'),
                    yaxis = cms.untracked.PSet(
                        edges = cms.untracked.vdouble(0.0, 0.0608, 0.1216, 0.1824, 0.2432, 
                            0.304, 0.3648, 0.4256, 0.4864, 0.5472, 
                            0.608, 0.608, 1.216, 1.824, 2.432, 
                            3.04, 3.648, 4.256, 4.864, 5.472, 
                            6.08, 6.688, 7.296, 7.904, 8.512, 
                            9.12, 9.728, 10.336, 10.944, 11.552, 
                            12.16, 12.768, 13.376, 13.984, 14.592, 
                            15.2, 15.808, 16.416, 17.024, 17.632, 
                            18.24, 18.848, 19.456, 20.064, 20.672, 
                            21.28, 21.888, 22.496, 23.104, 23.712, 
                            24.32, 24.928, 25.536, 26.144, 26.752, 
                            27.36, 27.968, 28.576, 29.184, 29.792, 
                            30.4, 31.008, 31.616, 32.224, 32.832, 
                            33.44, 34.048, 34.656, 35.264, 35.872, 
                            36.48, 37.088, 37.696, 38.304, 38.912, 
                            39.52, 40.128, 40.736, 41.344),
                        title = cms.untracked.string('event size (kB)')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size vs DCC')
                ),
                DCCSizeProf = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Mean and spread of the per-DCC data size.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('event size (kB)')
                    ),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT DCC event size')
                ),
                ZSMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1+ZS2 counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of ZS1 and ZS2 flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                HighIntPayload = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total data size from all high interest towers.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(3.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('event size (kB)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest payload%(suffix)s')
                ),
                EventSize = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of per-DCC data size.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(3.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('event size (kB)')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size%(suffix)s')
                ),
                FullReadoutMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower full readout counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy with FR flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                FlagCounterMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower flag counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of any SR flag.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                FullReadout = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Number of FR flags per event.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(200.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('number of towers')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags Number%(suffix)s')
                ),
                RUForcedMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT RU with forced SR counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of FORCED flag.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                )
            )
        ),
        OccupancyTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                recHitThreshold = cms.untracked.double(0.5),
                tpThreshold = cms.untracked.double(4.0)
            ),
            MEs = cms.untracked.PSet(
                TrendNTPDigi = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered TP digis'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the per-event number of TP digis with Et > 4.0 GeV.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                RecHitThr1D = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(500.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of filtered rec hits in event')
                ),
                Digi1D = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the number of digis per event.'),
                    otype = cms.untracked.string('Ecal2P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(3000.0),
                        nbins = cms.untracked.int32(100),
                        low = cms.untracked.double(0.0)
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of digis in event')
                ),
                DigiAll = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Digi occupancy.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                TPDigiThrProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection phi'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                RecHitThrProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection eta'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                RecHitProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection eta'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the occupancy of all rec hits.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                DigiProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection eta'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of digi occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                TrendNRecHitThr = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered recHits'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the per-event number of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                TPDigiThrAll = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy for TP digis with Et > 4.0 GeV.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                DCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT DCC entries'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Number of entries recoreded by each FED'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                TPDigiThrProjEta = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection eta'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjEta')
                ),
                TrendNDigi = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of digis'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Trend of the per-event number of digis.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Trend')
                ),
                Digi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string('Digi occupancy.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                DigiDCC = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT digi occupancy summary 1D'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('DCC digi occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                RecHitAll = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Rec hit occupancy.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                RecHitProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection phi'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the rec hit occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                RecHitThrProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection phi'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                DigiProjPhi = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection phi'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Projection of digi occupancy.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('ProjPhi')
                ),
                RecHitThrAll = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy for rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                )
            )
        ),
        TrigPrimTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                runOnEmul = cms.untracked.bool(True)
            ),
            MEs = cms.untracked.PSet(
                LowIntMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of low interest flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                HighIntMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of high interest flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                EtMaxEmul = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the maximum Et value within one emulated TP'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(256.0),
                        nbins = cms.untracked.int32(128),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('TP Et')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/Emulated/%(prefix)sTTT Et spectrum Emulated Digis max%(suffix)s')
                ),
                EtReal = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the trigger primitive Et.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(256.0),
                        nbins = cms.untracked.int32(128),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('TP Et')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et spectrum Real Digis%(suffix)s')
                ),
                FGEmulError = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulFineGrainVetoError %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                EtVsBx = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('Mean TP Et in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
                    yaxis = cms.untracked.PSet(
                        title = cms.untracked.string('TP Et')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(16.0),
                        nbins = cms.untracked.int32(16),
                        labels = cms.untracked.vstring('1', 
                            '271', 
                            '541', 
                            '892', 
                            '1162', 
                            '1432', 
                            '1783', 
                            '2053', 
                            '2323', 
                            '2674', 
                            '2944', 
                            '3214', 
                            '3446', 
                            '3490', 
                            '3491', 
                            '3565'),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('bunch crossing')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s')
                ),
                EtEmulError = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulError %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                MatchedIndex = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Counter for TP "timing" (= index withing the emulated TP whose Et matched that of the real TP)'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(6.0),
                        nbins = cms.untracked.int32(6),
                        labels = cms.untracked.vstring('no emul', 
                            '0', 
                            '1', 
                            '2', 
                            '3', 
                            '4'),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('TP index')
                    ),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('TriggerTower'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulMatch %(sm)s')
                ),
                EmulMaxIndex = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Distribution of the index of emulated TP with the highest Et value.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(5.5),
                        nbins = cms.untracked.int32(6),
                        labels = cms.untracked.vstring('no maximum', 
                            '0', 
                            '1', 
                            '2', 
                            '3', 
                            '4'),
                        low = cms.untracked.double(-0.5),
                        title = cms.untracked.string('TP index')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT max TP matching index%(suffix)s')
                ),
                MedIntMap = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Tower occupancy of medium interest flags.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                TTFlags = cms.untracked.PSet(
                    kind = cms.untracked.string('TH2F'),
                    description = cms.untracked.string('Distribution of the trigger tower flags.'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(7.5),
                        nbins = cms.untracked.int32(8),
                        labels = cms.untracked.vstring('0', 
                            '1', 
                            '2', 
                            '3', 
                            '4', 
                            '5', 
                            '6', 
                            '7'),
                        low = cms.untracked.double(-0.5),
                        title = cms.untracked.string('TT flag')
                    ),
                    otype = cms.untracked.string('Ecal3P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT Flags%(suffix)s')
                ),
                TTFMismatch = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT flag mismatch%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('TriggerTower')
                ),
                EtSummary = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
                    otype = cms.untracked.string('Ecal3P'),
                    zaxis = cms.untracked.PSet(
                        high = cms.untracked.double(256.0),
                        nbins = cms.untracked.int32(128),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('TP Et')
                    ),
                    btype = cms.untracked.string('TriggerTower'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Et trigger tower summary')
                ),
                EtRealMap = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile2D'),
                    description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
                    otype = cms.untracked.string('SM'),
                    zaxis = cms.untracked.PSet(
                        high = cms.untracked.double(256.0),
                        nbins = cms.untracked.int32(128),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('TP Et')
                    ),
                    btype = cms.untracked.string('TriggerTower'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et map Real Digis %(sm)s')
                ),
                OccVsBx = cms.untracked.PSet(
                    kind = cms.untracked.string('TProfile'),
                    description = cms.untracked.string('TP occupancy in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
                    otype = cms.untracked.string('Ecal3P'),
                    xaxis = cms.untracked.PSet(
                        high = cms.untracked.double(16.0),
                        nbins = cms.untracked.int32(16),
                        labels = cms.untracked.vstring('1', 
                            '271', 
                            '541', 
                            '892', 
                            '1162', 
                            '1432', 
                            '1783', 
                            '2053', 
                            '2323', 
                            '2674', 
                            '2944', 
                            '3214', 
                            '3446', 
                            '3490', 
                            '3491', 
                            '3565'),
                        low = cms.untracked.double(0.0),
                        title = cms.untracked.string('bunch crossing')
                    ),
                    btype = cms.untracked.string('User'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TP occupancy vs bx Real Digis%(suffix)s')
                )
            )
        ),
        IntegrityTask = cms.untracked.PSet(
            MEs = cms.untracked.PSet(
                Total = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Total number of integrity errors for each FED.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('DCC')
                ),
                BlockSize = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                ByLumi = cms.untracked.PSet(
                    kind = cms.untracked.string('TH1F'),
                    description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.'),
                    otype = cms.untracked.string('Ecal2P'),
                    btype = cms.untracked.string('DCC'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
                    perLumi = cms.untracked.bool(True)
                ),
                Gain = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                GainSwitch = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                TrendNErrors = cms.untracked.PSet(
                    path = cms.untracked.string('Ecal/Trends/IntegrityTask number of integrity errors'),
                    otype = cms.untracked.string('Ecal'),
                    description = cms.untracked.string('Trend of the number of integrity errors.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('Trend')
                ),
                ChId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                TowerId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
                    otype = cms.untracked.string('SM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                )
            )
        )
    ),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalPNDiodeMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string('EcalPNDiode Monitor Source'),
    resetInterval = cms.untracked.double(2.0),
    collectionTags = cms.untracked.PSet(
        TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
        EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
        TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
        EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
        PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
        EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
        EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
        Source = cms.untracked.InputTag("rawDataCollector"),
        MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
        MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
        EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
        TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
        EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
        EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
        EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
        EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
        EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
        EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
        EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EBSrFlag = cms.untracked.InputTag("ecalDigis"),
        BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
        EcalRawData = cms.untracked.InputTag("ecalDigis"),
        EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
        EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
        EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        EESrFlag = cms.untracked.InputTag("ecalDigis"),
        EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
    ),
    workers = cms.untracked.vstring('PNDiodeTask'),
    workerParameters = cms.untracked.PSet(
        PNDiodeTask = cms.untracked.PSet(
            MEs = cms.untracked.PSet(
                MEMBlockSize = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemSize/%(prefix)sIT MemSize %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMTowerId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemTTId/%(prefix)sIT MemTTId %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                OccupancySummary = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT PN digi occupancy summary'),
                    otype = cms.untracked.string('Ecal2P'),
                    description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMChId = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemChId/%(prefix)sIT MemChId %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
                    kind = cms.untracked.string('TH1F'),
                    btype = cms.untracked.string('Crystal')
                ),
                MEMGain = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemGain/%(prefix)sIT MemGain %(sm)s'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string(''),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('Crystal')
                ),
                Pedestal = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/PN/%(prefix)sPOT PN pedestal %(sm)s G16'),
                    otype = cms.untracked.string('SMMEM'),
                    description = cms.untracked.string('Presample mean of PN signals.'),
                    kind = cms.untracked.string('TProfile'),
                    btype = cms.untracked.string('Crystal')
                )
            )
        )
    ),
    verbosity = cms.untracked.int32(0),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalPedestalMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string('EcalPedestal Monitor Source'),
    resetInterval = cms.untracked.double(2.0),
    collectionTags = cms.untracked.PSet(
        TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
        EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
        TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
        EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
        PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
        EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
        EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
        Source = cms.untracked.InputTag("rawDataCollector"),
        MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
        MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
        EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
        TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
        EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
        EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
        EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
        EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
        EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
        EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
        EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EBSrFlag = cms.untracked.InputTag("ecalDigis"),
        BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
        EcalRawData = cms.untracked.InputTag("ecalDigis"),
        EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
        EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
        EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        EESrFlag = cms.untracked.InputTag("ecalDigis"),
        EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
    ),
    workers = cms.untracked.vstring('PedestalTask'),
    workerParameters = cms.untracked.PSet(
        PedestalTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                MGPAGainsPN = cms.untracked.vint32(16),
                MGPAGains = cms.untracked.vint32(12)
            ),
            MEs = cms.untracked.PSet(
                Pedestal = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('2D distribution of the mean pedestal.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s')
                ),
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT pedestal digi occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Channel occupancy in pedestal events.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                PNPedestal = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Pedestal distribution of the PN diodes.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s')
                )
            )
        )
    ),
    verbosity = cms.untracked.int32(0),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalTestPulseMonitorTask = cms.EDAnalyzer("EcalDQMonitorTask",
    moduleName = cms.untracked.string('EcalTestPulse Monitor Source'),
    resetInterval = cms.untracked.double(2.0),
    collectionTags = cms.untracked.PSet(
        TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
        EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
        TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
        EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
        PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
        EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
        EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
        Source = cms.untracked.InputTag("rawDataCollector"),
        MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
        MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
        EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
        TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
        EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
        EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
        EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
        EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
        EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
        EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
        EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
        EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
        EBSrFlag = cms.untracked.InputTag("ecalDigis"),
        BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
        EcalRawData = cms.untracked.InputTag("ecalDigis"),
        EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
        EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
        EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
        EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
        EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
        EESrFlag = cms.untracked.InputTag("ecalDigis"),
        EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
    ),
    workers = cms.untracked.vstring('TestPulseTask'),
    workerParameters = cms.untracked.PSet(
        TestPulseTask = cms.untracked.PSet(
            params = cms.untracked.PSet(
                MGPAGainsPN = cms.untracked.vint32(16),
                MGPAGains = cms.untracked.vint32(12)
            ),
            MEs = cms.untracked.PSet(
                Occupancy = cms.untracked.PSet(
                    path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT test pulse digi occupancy%(suffix)s'),
                    otype = cms.untracked.string('Ecal3P'),
                    description = cms.untracked.string('Occupancy in test pulse events.'),
                    kind = cms.untracked.string('TH2F'),
                    btype = cms.untracked.string('SuperCrystal')
                ),
                Shape = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Test pulse shape. One slice corresponds to one readout tower (5x5 crystals).'),
                    yaxis = cms.untracked.PSet(
                        high = cms.untracked.double(10.0),
                        nbins = cms.untracked.int32(10),
                        low = cms.untracked.double(0.0)
                    ),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('SuperCrystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT shape %(sm)s G%(gain)s')
                ),
                PNAmplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        pngain = cms.untracked.vint32(16)
                    ),
                    description = cms.untracked.string('Test pulse amplitude in the PN diodes.'),
                    kind = cms.untracked.string('TProfile'),
                    otype = cms.untracked.string('SMMEM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s')
                ),
                Amplitude = cms.untracked.PSet(
                    multi = cms.untracked.PSet(
                        gain = cms.untracked.vint32(12)
                    ),
                    description = cms.untracked.string('Test pulse amplitude.'),
                    kind = cms.untracked.string('TProfile2D'),
                    otype = cms.untracked.string('SM'),
                    btype = cms.untracked.string('Crystal'),
                    path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s')
                )
            )
        )
    ),
    verbosity = cms.untracked.int32(0),
    commonParameters = cms.untracked.PSet(
        willConvertToEDM = cms.untracked.bool(True),
        onlineMode = cms.untracked.bool(False)
    )
)


process.ecalRecoSequence = cms.Sequence(process.ecalGlobalUncalibRecHit+process.ecalDetIdToBeRecovered+process.ecalRecHit)


process.ecalPreRecoSequence = cms.Sequence(process.ecalDigis)


process.ecalLaserLedPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalLaserLedFilter+process.ecalRecoSequence+process.ecalLaserLedUncalibRecHit+process.ecalLaserLedMonitorTask+process.ecalPNDiodeMonitorTask)


process.ecalTestPulsePath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalTestPulseFilter+process.ecalRecoSequence+process.ecalTestPulseUncalibRecHit+process.ecalTestPulseMonitorTask+process.ecalPNDiodeMonitorTask)


process.ecalPedestalPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalPedestalFilter+process.ecalPedestalFilter+process.ecalRecoSequence+process.ecalPedestalMonitorTask+process.ecalPNDiodeMonitorTask)


process.ecalMonitorPath = cms.Path(process.ecalPreRecoSequence+process.ecalMonitorTask)


process.ecalClientPath = cms.Path(process.preScaler+process.ecalPreRecoSequence+process.ecalCalibrationFilter+process.ecalCalibrationFilter+process.ecalCalibMonitorClient)


process.dqmEndPath = cms.EndPath(process.dqmEnv)


process.dqmOutputPath = cms.EndPath(process.ecalMEFormatter+process.dqmSaver)


process.DQM = cms.Service("DQM",
    filter = cms.untracked.string(''),
    publishFrequency = cms.untracked.double(5.0),
    collectorHost = cms.untracked.string('dqm-prod-local.cms'),
    collectorPort = cms.untracked.int32(9090),
    debug = cms.untracked.bool(False)
)


process.DQMStore = cms.Service("DQMStore",
    verboseQT = cms.untracked.int32(0),
    enableMultiThread = cms.untracked.bool(False),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False),
    referenceFileName = cms.untracked.string('/dqmdata/dqm/reference/ecalcalib_reference.root')
)


process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('cerr')
)


process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'CASTOR', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)


process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")


process.CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")


process.CastorDbProducer = cms.ESProducer("CastorDbProducer")


process.CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP")


process.EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder")


process.EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService")


process.EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)


process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP",
    HcalReLabel = cms.PSet(
        RelabelRules = cms.untracked.PSet(
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            CorrectPhi = cms.untracked.bool(False)
        ),
        RelabelHits = cms.untracked.bool(False)
    )
)


process.SiStripRecHitMatcherESProducer = cms.ESProducer("SiStripRecHitMatcherESProducer",
    PreFilter = cms.bool(False),
    ComponentName = cms.string('StandardMatcher'),
    NSigmaInside = cms.double(3.0)
)


process.StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentType = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('StripCPEfromTrackAngle')
)


process.ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")


process.ecalSeverityLevel = cms.ESProducer("EcalSeverityLevelESProducer",
    dbstatusMask = cms.PSet(
        kRecovered = cms.vstring(),
        kGood = cms.vstring('kOk'),
        kTime = cms.vstring(),
        kWeird = cms.vstring(),
        kBad = cms.vstring('kNonRespondingIsolated', 
            'kDeadVFE', 
            'kDeadFE', 
            'kNoDataNoTP'),
        kProblematic = cms.vstring('kDAC', 
            'kNoLaser', 
            'kNoisy', 
            'kNNoisy', 
            'kNNNoisy', 
            'kNNNNoisy', 
            'kNNNNNoisy', 
            'kFixedG6', 
            'kFixedG1', 
            'kFixedG0')
    ),
    timeThresh = cms.double(2.0),
    flagMask = cms.PSet(
        kRecovered = cms.vstring('kLeadingEdgeRecovered', 
            'kTowerRecovered'),
        kGood = cms.vstring('kGood'),
        kTime = cms.vstring('kOutOfTime'),
        kWeird = cms.vstring('kWeird', 
            'kDiWeird'),
        kBad = cms.vstring('kFaultyHardware', 
            'kDead', 
            'kKilled'),
        kProblematic = cms.vstring('kPoorReco', 
            'kPoorCalib', 
            'kNoisy', 
            'kSaturated')
    )
)


process.hcalTopologyIdeal = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string(''),
    appendToDataLabel = cms.string(''),
    hcalTopologyConstants = cms.PSet(
        maxDepthHE = cms.int32(3),
        maxDepthHB = cms.int32(2),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    file = cms.untracked.string(''),
    dump = cms.untracked.vstring('')
)


process.siPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityFromDbRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiPixelDetVOffRcd'),
            tag = cms.string('')
        ))
)


process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer("SiStripBackPlaneCorrectionDepESProducer",
    LatencyRecord = cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        label = cms.untracked.string('')
    ),
    BackPlaneCorrectionDeconvMode = cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        label = cms.untracked.string('deconvolution')
    ),
    BackPlaneCorrectionPeakMode = cms.PSet(
        record = cms.string('SiStripBackPlaneCorrectionRcd'),
        label = cms.untracked.string('peak')
    )
)


process.siStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    printDebug = cms.untracked.bool(False),
    appendToDataLabel = cms.string(''),
    APVGain = cms.VPSet(cms.PSet(
        Record = cms.string('SiStripApvGainRcd'),
        NormalizationFactor = cms.untracked.double(1.0),
        Label = cms.untracked.string('')
    ), 
        cms.PSet(
            Record = cms.string('SiStripApvGain2Rcd'),
            NormalizationFactor = cms.untracked.double(1.0),
            Label = cms.untracked.string('')
        )),
    AutomaticNormalization = cms.bool(False)
)


process.siStripLorentzAngleDepESProducer = cms.ESProducer("SiStripLorentzAngleDepESProducer",
    LatencyRecord = cms.PSet(
        record = cms.string('SiStripLatencyRcd'),
        label = cms.untracked.string('')
    ),
    LorentzAngleDeconvMode = cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        label = cms.untracked.string('deconvolution')
    ),
    LorentzAnglePeakMode = cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        label = cms.untracked.string('peak')
    )
)


process.siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    appendToDataLabel = cms.string(''),
    PrintDebugOutput = cms.bool(False),
    ThresholdForReducedGranularity = cms.double(0.3),
    UseEmptyRunInfo = cms.bool(False),
    ReduceGranularity = cms.bool(False),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiStripDetCablingRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('RunInfoRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadFiberRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadModuleRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripBadStripRcd'),
            tag = cms.string('')
        ))
)


process.sistripconn = cms.ESProducer("SiStripConnectivity")


process.GlobalTag = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string(''),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(60),
        authenticationSystem = cms.untracked.int32(0),
        connectionRetrialPeriod = cms.untracked.int32(10)
    ),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalDQMChannelStatusRcd'),
        tag = cms.string('EcalDQMChannelStatus_v1_hlt'),
        connect = cms.untracked.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL')
    ), 
        cms.PSet(
            record = cms.string('EcalDQMTowerStatusRcd'),
            tag = cms.string('EcalDQMTowerStatus_v1_hlt'),
            connect = cms.untracked.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_34X_ECAL')
        )),
    connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG'),
    globaltag = cms.string('GR_R_71_V6::All')
)


process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/cmsTracker.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/CMSCommonData/data/cmsMuon.xml', 
        'Geometry/CMSCommonData/data/mgnt.xml', 
        'Geometry/CMSCommonData/data/beampipe.xml', 
        'Geometry/CMSCommonData/data/cmsBeam.xml', 
        'Geometry/CMSCommonData/data/muonMB.xml', 
        'Geometry/CMSCommonData/data/muonMagnet.xml', 
        'Geometry/TrackerCommonData/data/pixfwdMaterials.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x2.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x3.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x4.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanelBase.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanel.xml', 
        'Geometry/TrackerCommonData/data/pixfwdBlade.xml', 
        'Geometry/TrackerCommonData/data/pixfwdNipple.xml', 
        'Geometry/TrackerCommonData/data/pixfwdDisk.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
        'Geometry/TrackerCommonData/data/pixbarladder.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
        'Geometry/TrackerCommonData/data/pixbar.xml', 
        'Geometry/TrackerCommonData/data/tibtidcommonmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmodpar.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0a.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0b.xml', 
        'Geometry/TrackerCommonData/data/tibmodule2.xml', 
        'Geometry/TrackerCommonData/data/tibstringpar.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring0lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring0.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring1lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring1.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring2lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring2.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring3lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring3.xml', 
        'Geometry/TrackerCommonData/data/tiblayerpar.xml', 
        'Geometry/TrackerCommonData/data/tiblayer0.xml', 
        'Geometry/TrackerCommonData/data/tiblayer1.xml', 
        'Geometry/TrackerCommonData/data/tiblayer2.xml', 
        'Geometry/TrackerCommonData/data/tiblayer3.xml', 
        'Geometry/TrackerCommonData/data/tib.xml', 
        'Geometry/TrackerCommonData/data/tidmaterial.xml', 
        'Geometry/TrackerCommonData/data/tidmodpar.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule2.xml', 
        'Geometry/TrackerCommonData/data/tidringpar.xml', 
        'Geometry/TrackerCommonData/data/tidring0.xml', 
        'Geometry/TrackerCommonData/data/tidring0f.xml', 
        'Geometry/TrackerCommonData/data/tidring0b.xml', 
        'Geometry/TrackerCommonData/data/tidring1.xml', 
        'Geometry/TrackerCommonData/data/tidring1f.xml', 
        'Geometry/TrackerCommonData/data/tidring1b.xml', 
        'Geometry/TrackerCommonData/data/tidring2.xml', 
        'Geometry/TrackerCommonData/data/tid.xml', 
        'Geometry/TrackerCommonData/data/tidf.xml', 
        'Geometry/TrackerCommonData/data/tidb.xml', 
        'Geometry/TrackerCommonData/data/tibtidservices.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesf.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesb.xml', 
        'Geometry/TrackerCommonData/data/tobmaterial.xml', 
        'Geometry/TrackerCommonData/data/tobmodpar.xml', 
        'Geometry/TrackerCommonData/data/tobmodule0.xml', 
        'Geometry/TrackerCommonData/data/tobmodule2.xml', 
        'Geometry/TrackerCommonData/data/tobmodule4.xml', 
        'Geometry/TrackerCommonData/data/tobrodpar.xml', 
        'Geometry/TrackerCommonData/data/tobrod0c.xml', 
        'Geometry/TrackerCommonData/data/tobrod0l.xml', 
        'Geometry/TrackerCommonData/data/tobrod0h.xml', 
        'Geometry/TrackerCommonData/data/tobrod0.xml', 
        'Geometry/TrackerCommonData/data/tobrod1l.xml', 
        'Geometry/TrackerCommonData/data/tobrod1h.xml', 
        'Geometry/TrackerCommonData/data/tobrod1.xml', 
        'Geometry/TrackerCommonData/data/tobrod2c.xml', 
        'Geometry/TrackerCommonData/data/tobrod2l.xml', 
        'Geometry/TrackerCommonData/data/tobrod2h.xml', 
        'Geometry/TrackerCommonData/data/tobrod2.xml', 
        'Geometry/TrackerCommonData/data/tobrod3l.xml', 
        'Geometry/TrackerCommonData/data/tobrod3h.xml', 
        'Geometry/TrackerCommonData/data/tobrod3.xml', 
        'Geometry/TrackerCommonData/data/tobrod4c.xml', 
        'Geometry/TrackerCommonData/data/tobrod4l.xml', 
        'Geometry/TrackerCommonData/data/tobrod4h.xml', 
        'Geometry/TrackerCommonData/data/tobrod4.xml', 
        'Geometry/TrackerCommonData/data/tobrod5l.xml', 
        'Geometry/TrackerCommonData/data/tobrod5h.xml', 
        'Geometry/TrackerCommonData/data/tobrod5.xml', 
        'Geometry/TrackerCommonData/data/tob.xml', 
        'Geometry/TrackerCommonData/data/tecmaterial.xml', 
        'Geometry/TrackerCommonData/data/tecmodpar.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule2.xml', 
        'Geometry/TrackerCommonData/data/tecmodule3.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule5.xml', 
        'Geometry/TrackerCommonData/data/tecmodule6.xml', 
        'Geometry/TrackerCommonData/data/tecpetpar.xml', 
        'Geometry/TrackerCommonData/data/tecring0.xml', 
        'Geometry/TrackerCommonData/data/tecring1.xml', 
        'Geometry/TrackerCommonData/data/tecring2.xml', 
        'Geometry/TrackerCommonData/data/tecring3.xml', 
        'Geometry/TrackerCommonData/data/tecring4.xml', 
        'Geometry/TrackerCommonData/data/tecring5.xml', 
        'Geometry/TrackerCommonData/data/tecring6.xml', 
        'Geometry/TrackerCommonData/data/tecring0f.xml', 
        'Geometry/TrackerCommonData/data/tecring1f.xml', 
        'Geometry/TrackerCommonData/data/tecring2f.xml', 
        'Geometry/TrackerCommonData/data/tecring3f.xml', 
        'Geometry/TrackerCommonData/data/tecring4f.xml', 
        'Geometry/TrackerCommonData/data/tecring5f.xml', 
        'Geometry/TrackerCommonData/data/tecring6f.xml', 
        'Geometry/TrackerCommonData/data/tecring0b.xml', 
        'Geometry/TrackerCommonData/data/tecring1b.xml', 
        'Geometry/TrackerCommonData/data/tecring2b.xml', 
        'Geometry/TrackerCommonData/data/tecring3b.xml', 
        'Geometry/TrackerCommonData/data/tecring4b.xml', 
        'Geometry/TrackerCommonData/data/tecring5b.xml', 
        'Geometry/TrackerCommonData/data/tecring6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetalf.xml', 
        'Geometry/TrackerCommonData/data/tecpetalb.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8b.xml', 
        'Geometry/TrackerCommonData/data/tecwheel.xml', 
        'Geometry/TrackerCommonData/data/tecwheela.xml', 
        'Geometry/TrackerCommonData/data/tecwheelb.xml', 
        'Geometry/TrackerCommonData/data/tecwheelc.xml', 
        'Geometry/TrackerCommonData/data/tecwheeld.xml', 
        'Geometry/TrackerCommonData/data/tecwheel6.xml', 
        'Geometry/TrackerCommonData/data/tecservices.xml', 
        'Geometry/TrackerCommonData/data/tecbackplate.xml', 
        'Geometry/TrackerCommonData/data/tec.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml', 
        'Geometry/TrackerCommonData/data/tracker.xml', 
        'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
        'Geometry/TrackerCommonData/data/trackerpixfwd.xml', 
        'Geometry/TrackerCommonData/data/trackertibtidservices.xml', 
        'Geometry/TrackerCommonData/data/trackertib.xml', 
        'Geometry/TrackerCommonData/data/trackertid.xml', 
        'Geometry/TrackerCommonData/data/trackertob.xml', 
        'Geometry/TrackerCommonData/data/trackertec.xml', 
        'Geometry/TrackerCommonData/data/trackerbulkhead.xml', 
        'Geometry/TrackerCommonData/data/trackerother.xml', 
        'Geometry/EcalCommonData/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/EcalCommonData/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eefixed.xml', 
        'Geometry/EcalCommonData/data/eehier.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/EcalCommonData/data/escon.xml', 
        'Geometry/EcalCommonData/data/esalgo.xml', 
        'Geometry/EcalCommonData/data/eeF.xml', 
        'Geometry/EcalCommonData/data/eeB.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalCommonData/data/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/average/hcalforwardmaterial.xml', 
        'Geometry/MuonCommonData/data/mbCommon.xml', 
        'Geometry/MuonCommonData/data/mb1.xml', 
        'Geometry/MuonCommonData/data/mb2.xml', 
        'Geometry/MuonCommonData/data/mb3.xml', 
        'Geometry/MuonCommonData/data/mb4.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonCommonData/data/mf.xml', 
        'Geometry/ForwardCommonData/data/forward.xml', 
        'Geometry/ForwardCommonData/data/bundle/forwardshield.xml', 
        'Geometry/ForwardCommonData/data/brmrotations.xml', 
        'Geometry/ForwardCommonData/data/brm.xml', 
        'Geometry/ForwardCommonData/data/totemMaterials.xml', 
        'Geometry/ForwardCommonData/data/totemRotations.xml', 
        'Geometry/ForwardCommonData/data/totemt1.xml', 
        'Geometry/ForwardCommonData/data/totemt2.xml', 
        'Geometry/ForwardCommonData/data/ionpump.xml', 
        'Geometry/MuonCommonData/data/muonNumbering.xml', 
        'Geometry/TrackerCommonData/data/trackerStructureTopology.xml', 
        'Geometry/TrackerSimData/data/trackersens.xml', 
        'Geometry/TrackerRecoData/data/trackerRecoMaterial.xml', 
        'Geometry/EcalSimData/data/ecalsens.xml', 
        'Geometry/HcalCommonData/data/hcalsenspmf.xml', 
        'Geometry/HcalSimData/data/hf.xml', 
        'Geometry/HcalSimData/data/hfpmt.xml', 
        'Geometry/HcalSimData/data/hffibrebundle.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/MuonSimData/data/muonSens.xml', 
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml', 
        'Geometry/ForwardCommonData/data/brmsens.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        'Geometry/EcalSimData/data/ESProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml', 
        'Geometry/MuonSimData/data/muonProdCuts.xml', 
        'Geometry/ForwardSimData/data/ForwardShieldProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


process.eegeom = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalMappingRcd'),
    firstValid = cms.vuint32(1)
)


process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    HcalReLabel = cms.PSet(
        RelabelRules = cms.untracked.PSet(
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            CorrectPhi = cms.untracked.bool(False)
        ),
        RelabelHits = cms.untracked.bool(False)
    ),
    HERecalibration = cms.bool(False),
    toGet = cms.untracked.vstring('GainWidths'),
    GainWidthsForTrigPrims = cms.bool(False),
    HEreCalibCutoff = cms.double(20.0),
    HFRecalibration = cms.bool(False),
    iLumi = cms.double(-1.0),
    hcalTopologyConstants = cms.PSet(
        maxDepthHE = cms.int32(3),
        maxDepthHB = cms.int32(2),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.essourceEcalSev = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalSeverityLevelAlgoRcd'),
    firstValid = cms.vuint32(1)
)


process.prefer("es_hardcode")

process.prefer("GlobalTag")

process.ecalCalibrationSummaryClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        Laser = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT%(suffix)s laser quality summary L%(wl)s')
        ),
        LaserPN = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Summary of the laser data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0, 800.0, 800.0 for laser 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT PN laser quality summary L%(wl)s')
        ),
        LedPN = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Summary of the led data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0 for led 1 and 2 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EEMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT PN led quality summary L%(wl)s')
        ),
        PedestalPN = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Summary of the pedestal quality for PN diodes. A channel is red if the pedestal mean is off from 750.0 by 100.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT PN pedestal quality G%(pngain)s summary')
        ),
        PNIntegrity = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary'),
            otype = cms.untracked.string('MEM2P'),
            description = cms.untracked.string('Summary of the data integrity in PN channels. A channel is red if more than 1.0% of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Led = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EE2P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT%(suffix)s led quality summary L%(wl)s')
        ),
        Pedestal = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT pedestal quality G%(gain)s summary%(suffix)s')
        ),
        TestPulsePN = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Summary of test pulse data quality for PN diodes. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 12.5, 200.0 and 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary')
        ),
        TestPulse = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary')
        )
    ),
    params = cms.untracked.PSet(
        testPulseMGPAGains = cms.untracked.vint32(12),
        laserWavelengths = cms.untracked.vint32(1, 2, 3),
        pedestalMGPAGainsPN = cms.untracked.vint32(16),
        pedestalMGPAGains = cms.untracked.vint32(12),
        testPulseMGPAGainsPN = cms.untracked.vint32(16),
        ledWavelengths = cms.untracked.vint32(1, 2),
        activeSources = cms.untracked.vstring('Laser', 
            'Led', 
            'TestPulse')
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global calibration quality%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the calibration data quality. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s PN global quality'),
            otype = cms.untracked.string('MEM2P'),
            description = cms.untracked.string('Summary of the calibration data quality for the PN diodes. Channel is red if it is red in any of the Laser 3, Led 1 and 2, Pedestal gain 12, and Test Pulse gain 12 quality summary.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalClusterTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        swissCrossMaxThreshold = cms.untracked.double(3.0),
        L1MuGMTReadoutCollectionTag = cms.untracked.InputTag("gtDigis"),
        egTriggerAlgos = cms.untracked.vstring('L1_SingleEG2', 
            'L1_SingleEG5', 
            'L1_SingleEG8', 
            'L1_SingleEG10', 
            'L1_SingleEG12', 
            'L1_SingleEG15', 
            'L1_SingleEG20', 
            'L1_SingleEG25', 
            'L1_DoubleNoIsoEG_BTB_tight', 
            'L1_DoubleNoIsoEG_BTB_loose', 
            'L1_DoubleNoIsoEGTopBottom', 
            'L1_DoubleNoIsoEGTopBottomCen', 
            'L1_DoubleNoIsoEGTopBottomCen2', 
            'L1_DoubleNoIsoEGTopBottomCenVert'),
        L1GlobalTriggerReadoutRecordTag = cms.untracked.InputTag("gtDigis"),
        doExtra = cms.untracked.bool(True),
        energyThreshold = cms.untracked.double(2.0)
    ),
    MEs = cms.untracked.PSet(
        BCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection phi%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        TrendNBC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of basic clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the number of basic clusters per event in EB/EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        BCSizeMapProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection eta%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('ProjEta')
        ),
        BCSize = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the basic cluster size (number of crystals).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size')
        ),
        BCEtMapProjEta = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection eta%(suffix)s')
        ),
        SCR9 = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of E_seed / E_3x3 of the super clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.2),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC R9')
        ),
        TrendSCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of super clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the mean size (number of crystals) of the super clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        TrendNSC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of super clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the number of super clusters per event in EB/EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        SCNum = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of super clusters per event in EB/EE.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC number')
        ),
        SCSeedOccupancyHighE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (high energy clusters) %(supercrystal)s binned'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters with energy > 2.0 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        SCE = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Super cluster energy distribution.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy')
        ),
        SCSizeVsEnergy = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Mean SC size in crystals as a function of the SC energy.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC size (crystal) vs energy (GeV)')
        ),
        SCSeedOccupancyTrig = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (%(trig)s triggered) %(supercrystal)s binned')
        ),
        SCSeedEnergy = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Energy distribution of the crystals that seeded super clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed crystal energy')
        ),
        SCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_eta'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Supercluster eta.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        SCSeedTimeMapTrigEx = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Mean timing of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing map%(suffix)s (%(trig)s exclusive triggered) %(supercrystal)s binned')
        ),
        BCE = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Basic cluster energy distribution.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy')
        ),
        BCSizeMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('2D distribution of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        SCNcrystals = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the super cluster size (number of crystals).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size (crystal)')
        ),
        BCNum = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of basic clusters per event.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number')
        ),
        SCSeedTimeTrigEx = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                trig = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC')
            ),
            description = cms.untracked.string('Timing distribution of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing (%(trig)s exclusive triggered)')
        ),
        SCSwissCross = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Swiss cross for SC maximum-energy crystal.'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBRecoSummary/superClusters_EB_E1oE4')
        ),
        SingleCrystalCluster = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC single crystal cluster seed occupancy map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the occurrence of super clusters with only one constituent'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Triggers = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Counter for the trigger categories'),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(5),
                labels = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('triggers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE triggers')
        ),
        BCSizeMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection phi%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('ProjPhi')
        ),
        BCOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Basic cluster occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        BCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection eta%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        SCNBCs = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the super cluster size (number of basic clusters)'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.0),
                nbins = cms.untracked.int32(15),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size')
        ),
        BCEtMapProjPhi = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection phi%(suffix)s')
        ),
        SCSeedOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed occupancy map%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        SCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_phi'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Supercluster phi.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        BCEMapProjEta = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection eta%(suffix)s')
        ),
        SCELow = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Energy distribution of the super clusters (low scale).'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy (low scale)')
        ),
        TrendBCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of basic clusters'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the mean size of the basic clusters.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        ExclusiveTriggers = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Counter for the trigger categories'),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(5),
                labels = cms.untracked.vstring('ECAL', 
                    'HCAL', 
                    'CSC', 
                    'DT', 
                    'RPC'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('triggers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE exclusive triggers')
        ),
        BCEMapProjPhi = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection phi%(suffix)s')
        ),
        SCClusterVsSeed = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Relation between super cluster energy and its seed crystal energy.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy vs seed crystal energy')
        ),
        BCEMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean energy of the basic clusters.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy map%(suffix)s')
        )
    )
)

process.ecalCommonParams = cms.untracked.PSet(
    willConvertToEDM = cms.untracked.bool(True),
    onlineMode = cms.untracked.bool(False)
)

process.ecalDQMCollectionTags = cms.untracked.PSet(
    TowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    EEUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    TrigPrimDigi = cms.untracked.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEE"),
    PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
    EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
    EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
    EBRecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
    Source = cms.untracked.InputTag("rawDataCollector"),
    MEMGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
    MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
    EEDigi = cms.untracked.InputTag("ecalDigis","eeDigis"),
    TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
    EBDigi = cms.untracked.InputTag("ecalDigis","ebDigis"),
    EBUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
    EEChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EEGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit","EcalUncalibRecHitsEB"),
    EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    MEMChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
    EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
    EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
    EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
    EBChIdErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EBSrFlag = cms.untracked.InputTag("ecalDigis"),
    BlockSizeErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    EcalRawData = cms.untracked.InputTag("ecalDigis"),
    EERecHit = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
    EBGainErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
    EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    EESrFlag = cms.untracked.InputTag("ecalDigis"),
    EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE")
)

process.ecalEnergyTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        isPhysicsRun = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        HitMapAll = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s energy summary')
        ),
        HitAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rec hit energy distribution.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit spectrum%(suffix)s')
        ),
        Hit = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rec hit energy distribution.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT energy spectrum %(sm)s')
        ),
        HitMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit energy.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit energy %(sm)s')
        )
    )
)

process.ecalIntegrityClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        BlockSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Gain = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        GainSwitch = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        ChId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        TowerId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    ),
    params = cms.untracked.PSet(
        errFractionThreshold = cms.untracked.double(0.01)
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityClient/%(prefix)sIT data integrity quality %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalIntegrityTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        Total = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT integrity quality errors summary'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Total number of integrity errors for each FED.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        BlockSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTBlockSize/%(prefix)sIT TTBlockSize %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        Gain = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/Gain/%(prefix)sIT gain %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        GainSwitch = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/GainSwitch/%(prefix)sIT gain switch %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        TrendNErrors = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/IntegrityTask number of integrity errors'),
            otype = cms.untracked.string('Ecal'),
            description = cms.untracked.string('Trend of the number of integrity errors.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('Trend')
        ),
        ChId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/ChId/%(prefix)sIT ChId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        TowerId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/TTId/%(prefix)sIT TTId %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    )
)

process.ecalLaserClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        Timing = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean laser timing. Z scale is in LHC clocks. Due to the difference in pulse shape between laser and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT timing %(sm)s L%(wl)s')
        ),
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Mean laser pulse amplitude in the PN diodes. In general, a PN channel is filled only when a laser pulse was observed in the crystals that are associated to the diode. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/PN/Gain16/%(prefix)sLT PNs amplitude %(sm)s G16 L%(wl)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude %(sm)s L%(wl)s')
        )
    ),
    params = cms.untracked.PSet(
        expectedTiming = cms.untracked.vdouble(4.2, 4.2, 4.2, 4.2),
        tolerancePNAmp = cms.untracked.double(0.1),
        toleranceTimRMS = cms.untracked.double(0.4),
        forwardFactor = cms.untracked.double(0.5),
        toleranceTiming = cms.untracked.double(0.5),
        expectedPNAmplitude = cms.untracked.vdouble(800.0, 800.0, 800.0, 800.0),
        tolerancePNRMSRatio = cms.untracked.double(1.0),
        toleranceAmpRMSRatio = cms.untracked.double(0.3),
        minChannelEntries = cms.untracked.int32(3),
        toleranceAmplitude = cms.untracked.double(0.1),
        expectedAmplitude = cms.untracked.vdouble(1700.0, 1300.0, 1700.0, 1700.0)
    ),
    MEs = cms.untracked.PSet(
        TimingRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Distribution of the timing RMS in each crystal channel. X scale is in LHC clocks. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(0.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('rms (clock)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms L%(wl)s %(sm)s')
        ),
        TimingMean = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Distribution of the timing in each crystal channel. X scale is in LHC clocks. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(3.5),
                title = cms.untracked.string('time (clock)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing mean L%(wl)s %(sm)s')
        ),
        PNQualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Summary of the laser data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0, 800.0, 800.0 for laser 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT PN laser quality summary L%(wl)s')
        ),
        TimingRMSMap = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the laser timing RMS. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (clock)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser timing rms map L%(wl)s')
        ),
        AmplitudeMean = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(4096.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude L%(wl)s %(sm)s')
        ),
        QualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sLT%(suffix)s laser quality summary L%(wl)s')
        ),
        Quality = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Summary of the laser data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.3, or if mean timing is off from expected by 0.5. Expected amplitudes and timings are 1700.0, 1300.0, 1700.0, 1700.0 and 4.2, 4.2, 4.2, 4.2 for lasers 1, 2, 3, and 4 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT laser quality L%(wl)s %(sm)s')
        ),
        AmplitudeRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ADC counts)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserClient/%(prefix)sLT amplitude rms L%(wl)s')
        )
    )
)

process.ecalLaserTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        emptyLSLimit = cms.untracked.int32(3),
        laserWavelengths = cms.untracked.vint32(1, 2, 3)
    ),
    MEs = cms.untracked.PSet(
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Mean laser pulse amplitude in the PN diodes. In general, a PN channel is filled only when a laser pulse was observed in the crystals that are associated to the diode. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/PN/Gain16/%(prefix)sLT PNs amplitude %(sm)s G16 L%(wl)s')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT laser digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Laser signal digi occupancy. Channels are filled whenever the DCC event type is set to LASER.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Timing = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean laser timing. Z scale is in LHC clocks. Due to the difference in pulse shape between laser and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT timing %(sm)s L%(wl)s')
        ),
        Shape = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('Laser mean pulse shape. One slice corresponds to one readout tower (5x5 crystals). In general, a slice is filled only when a laser pulse was observed in the tower. When no laser signal was observed for longer than 3 lumi sections, the slices start to get filled with 0 amplitude, causing the shape to flatten.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(10),
                low = cms.untracked.double(0.0)
            ),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT shape %(sm)s L%(wl)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude %(sm)s L%(wl)s')
        ),
        AOverP = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean APD/PN value (event mean of per-event ratio).'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude over PN %(sm)s L%(wl)s')
        ),
        SignalRate = cms.untracked.PSet(
            multi = cms.untracked.int32(4),
            description = cms.untracked.string('Fraction of laser events with measurable laser pulse.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT signal rate L%(wl)s')
        ),
        AmplitudeSummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2, 3)
            ),
            description = cms.untracked.string('2D distribution of the mean laser amplitude. In general, a channel is filled only when a laser pulse was observed in it. When no laser signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sLaserTask/Laser%(wl)s/%(prefix)sLT amplitude map L%(wl)s%(suffix)s')
        )
    )
)

process.ecalLedClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        Timing = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the mean led timing. Z scale is in LHC clocks. Due to the difference in pulse shape between led and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s')
        ),
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Mean led pulse amplitude in the PN diodes. In general, a PN channel is filled only when a led pulse was observed in the crystals that are associated to the diode. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EESMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s')
        )
    ),
    params = cms.untracked.PSet(
        expectedTiming = cms.untracked.vdouble(4.4, 4.5),
        tolerancePNAmp = cms.untracked.double(0.1),
        toleranceTimRMS = cms.untracked.double(25.0),
        forwardFactor = cms.untracked.double(0.5),
        toleranceTiming = cms.untracked.double(1.0),
        expectedPNAmplitude = cms.untracked.vdouble(800.0, 800.0),
        tolerancePNRMSRatio = cms.untracked.double(1.0),
        toleranceAmpRMSRatio = cms.untracked.double(0.5),
        minChannelEntries = cms.untracked.int32(3),
        toleranceAmplitude = cms.untracked.double(0.1),
        expectedAmplitude = cms.untracked.vdouble(200.0, 10.0)
    ),
    MEs = cms.untracked.PSet(
        TimingMean = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Distribution of the timing in each crystal channel. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EESM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(3.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led timing L%(wl)s %(sm)s')
        ),
        PNQualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Summary of the led data quality in the PN diodes. A channel is red if mean / expected < 0.1 or RMS / expected > 1.0. Expected amplitudes are 800.0, 800.0 for led 1 and 2 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EEMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT PN led quality summary L%(wl)s')
        ),
        TimingRMSMap = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the led timing RMS. Z scale is in LHC clocks. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EE'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT timing RMS L%(wl)s')
        ),
        AmplitudeMean = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Distribution of the mean amplitude seen in each crystal. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EESM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(400.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude L%(wl)s %(sm)s')
        ),
        QualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EE2P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('EcalEndcap/EESummaryClient/EELDT%(suffix)s led quality summary L%(wl)s')
        ),
        Quality = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Summary of the led data quality. A channel is red either if mean / expected < 0.1, or if RMS / expected > 0.5, or if mean timing is off from expected by 1.0. Expected amplitudes and timings are 200.0, 10.0 and 4.4, 4.5 for leds 1 and 2 respectively. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT led quality L%(wl)s %(sm)s')
        ),
        AmplitudeRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EE'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedClient/EELDT amplitude RMS L%(wl)s')
        )
    )
)

process.ecalLedTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        emptyLSLimit = cms.untracked.int32(3),
        ledWavelengths = cms.untracked.vint32(1, 2)
    ),
    MEs = cms.untracked.PSet(
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Mean led pulse amplitude in the PN diodes. In general, a PN channel is filled only when a led pulse was observed in the crystals that are associated to the diode. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EESMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EEOccupancyTask/EEOT led digi occupancy%(suffix)s'),
            otype = cms.untracked.string('EE2P'),
            description = cms.untracked.string('Led signal digi occupancy. Channels are filled whenever the DCC event type is set to LED.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Timing = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the mean led timing. Z scale is in LHC clocks. Due to the difference in pulse shape between led and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s')
        ),
        Shape = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('Led mean pulse shape. One slice corresponds to one readout tower (5x5 crystals). In general, a slice is filled only when a led pulse was observed in the tower. When no led signal was observed for longer than 3 lumi sections, the slices start to get filled with 0 amplitude, causing the shape to flatten.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(10),
                low = cms.untracked.double(0.0)
            ),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT shape %(sm)s L%(wl)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s')
        ),
        AOverP = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = cms.untracked.vint32(1, 2)
            ),
            description = cms.untracked.string('2D distribution of the mean APD/PN value (event mean of per-event ratio).'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude over PN %(sm)s L%(wl)s')
        ),
        SignalRate = cms.untracked.PSet(
            multi = cms.untracked.int32(2),
            description = cms.untracked.string('Fraction of laser events with measurable led pulse.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EE'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT signal rate L%(wl)s')
        ),
        AmplitudeSummary = cms.untracked.PSet(
            multi = cms.untracked.int32(4),
            description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than 3 lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EE2P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude map L%(wl)s%(suffix)s')
        )
    )
)

process.ecalOccupancyClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        RecHitThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        DigiAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        TPDigiThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        )
    ),
    params = cms.untracked.PSet(
        deviationThreshold = cms.untracked.double(100.0),
        minHits = cms.untracked.int32(20)
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the hot cell monitor. A channel is red if it has more than 100.0 times more entries than phi-ring mean in either digi, rec hit (filtered), or TP digi (filtered). Channels with less than 20 entries are not considered. Channel names of the hot cells are available in (Top) / Ecal / Errors / HotCells.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    )
)

process.ecalOccupancyTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        recHitThreshold = cms.untracked.double(0.5),
        tpThreshold = cms.untracked.double(4.0)
    ),
    MEs = cms.untracked.PSet(
        TrendNTPDigi = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered TP digis'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        RecHitThr1D = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(500.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of filtered rec hits in event')
        ),
        Digi1D = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the number of digis per event.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3000.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT number of digis in event')
        ),
        DigiAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        TPDigiThrProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        RecHitThrProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        RecHitProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of all rec hits.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        DigiProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        TrendNRecHitThr = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of filtered recHits'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        TPDigiThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        DCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT DCC entries'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of entries recoreded by each FED'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        TPDigiThrProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT TP digi thr occupancy%(suffix)s projection eta'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of TP digis with Et > 4.0 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjEta')
        ),
        TrendNDigi = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/OccupancyTask %(prefix)s number of digis'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of the per-event number of digis.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        Digi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Digi occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        DigiDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT digi occupancy summary 1D'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('DCC digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RecHitAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Rec hit occupancy.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        RecHitProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the rec hit occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        RecHitThrProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of the occupancy of rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        DigiProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT digi occupancy%(suffix)s projection phi'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Projection of digi occupancy.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('ProjPhi')
        ),
        RecHitThrAll = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT rec hit thr occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy for rec hits with GOOD reconstruction flag and E > 0.5 GeV.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    )
)

process.ecalPNDiodeTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        MEMBlockSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemSize/%(prefix)sIT MemSize %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMTowerId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemTTId/%(prefix)sIT MemTTId %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        OccupancySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT PN digi occupancy summary'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMChId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemChId/%(prefix)sIT MemChId %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMGain = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemGain/%(prefix)sIT MemGain %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/PN/%(prefix)sPOT PN pedestal %(sm)s G16'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string('Presample mean of PN signals.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalPNIntegrityClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        MEMBlockSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemSize/%(prefix)sIT MemSize %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMChId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemChId/%(prefix)sIT MemChId %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMTowerId = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemTTId/%(prefix)sIT MemTTId %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT MEM digi occupancy %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string('Occupancy of PN digis in calibration events.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('Crystal')
        ),
        MEMGain = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/MemGain/%(prefix)sIT MemGain %(sm)s'),
            otype = cms.untracked.string('SMMEM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    ),
    params = cms.untracked.PSet(
        errFractionThreshold = cms.untracked.double(0.01)
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT PN integrity quality summary'),
            otype = cms.untracked.string('MEM2P'),
            description = cms.untracked.string('Summary of the data integrity in PN channels. A channel is red if more than 1.0% of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalPedestalClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('2D distribution of the mean pedestal.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s')
        ),
        PNPedestal = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Pedestal distribution of the PN diodes.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s')
        )
    ),
    params = cms.untracked.PSet(
        toleranceRMS = cms.untracked.vdouble(1.0, 1.2, 2.0),
        expectedMean = cms.untracked.double(200.0),
        tolerancePNRMS = cms.untracked.vdouble(20.0, 20.0),
        toleranceMean = cms.untracked.double(25.0),
        tolerancePNMean = cms.untracked.double(100.0),
        expectedPNMean = cms.untracked.double(750.0),
        minChannelEntries = cms.untracked.int32(3)
    ),
    MEs = cms.untracked.PSet(
        RMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Distribution of the pedestal RMS for each crystal channel. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal rms G%(gain)s %(sm)s')
        ),
        PNRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Distribution of the pedestal RMS for each PN channel. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SMMEM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPDT PNs pedestal rms %(sm)s G%(pngain)s')
        ),
        PNQualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Summary of the pedestal quality for PN diodes. A channel is red if the pedestal mean is off from 750.0 by 100.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT PN pedestal quality G%(pngain)s summary')
        ),
        QualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT pedestal quality G%(gain)s summary%(suffix)s')
        ),
        Quality = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from 200.0 by 25.0 or if the pedestal RMS is greater than threshold. RMS thresholds are 1.0, 1.2, 2.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal quality G%(gain)s %(sm)s')
        ),
        Mean = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Distribution of pedestal mean in each channel. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(230.0),
                nbins = cms.untracked.int32(120),
                low = cms.untracked.double(170.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal mean G%(gain)s %(sm)s')
        )
    )
)

process.ecalPedestalTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        MGPAGainsPN = cms.untracked.vint32(16),
        MGPAGains = cms.untracked.vint32(12)
    ),
    MEs = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('2D distribution of the mean pedestal.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/Gain%(gain)s/%(prefix)sPT pedestal %(sm)s G%(gain)s')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT pedestal digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Channel occupancy in pedestal events.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        PNPedestal = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Pedestal distribution of the PN diodes.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalTask/PN/Gain%(pngain)s/%(prefix)sPDT PNs pedestal %(sm)s G%(pngain)s')
        )
    )
)

process.ecalPresampleClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('2D distribution of mean presample value.'),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal')
        )
    ),
    params = cms.untracked.PSet(
        toleranceRMSFwd = cms.untracked.double(6.0),
        toleranceRMS = cms.untracked.double(3.0),
        toleranceMean = cms.untracked.double(25.0),
        minChannelEntries = cms.untracked.int32(6),
        expectedMean = cms.untracked.double(200.0)
    ),
    MEs = cms.untracked.PSet(
        RMS = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the presample RMS of each channel. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms G12 %(sm)s')
        ),
        TrendRMS = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal rms max'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of presample RMS averaged over all channels in EB / EE.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        RMSMap = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal rms map G12 %(sm)s')
        ),
        TrendMean = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/PresampleClient %(prefix)s pedestal mean max - min'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Trend of presample spread in EB / EE. Y value indicates the difference between maximum and minimum presample mean values within the subdetector.'),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Trend')
        ),
        RMSMapAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('2D distribution of the presample RMS. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('RMS')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal G12 RMS map')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal quality G12 %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        ErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT pedestal quality errors summary G12'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Counter of channels flagged as bad in the quality summary'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        Mean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('1D distribution of the mean presample value in each crystal. Channels with entries less than 6 are not considered.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(230.0),
                nbins = cms.untracked.int32(120),
                low = cms.untracked.double(170.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineClient/%(prefix)sPOT pedestal mean G12 %(sm)s')
        )
    )
)

process.ecalPresampleTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        nSamples = cms.untracked.int32(3),
        pulseMaxPosition = cms.untracked.int32(5)
    ),
    MEs = cms.untracked.PSet(
        Pedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalOnlineTask/Gain12/%(prefix)sPOT pedestal %(sm)s G12'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('2D distribution of mean presample value.'),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalRawDataClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        FEStatus = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('FE status counter.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('ENABLED', 
                    'DISABLED', 
                    'TIMEOUT', 
                    'HEADERERROR', 
                    'CHANNELID', 
                    'LINKERROR', 
                    'BLOCKSIZE', 
                    'SUPPRESSED', 
                    'FIFOFULL', 
                    'L1ADESYNC', 
                    'BXDESYNC', 
                    'L1ABXDESYNC', 
                    'FIFOFULLL1ADESYNC', 
                    'HPARITY', 
                    'VPARITY', 
                    'FORCEDZS'),
                low = cms.untracked.double(-0.5)
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s')
        ),
        L1ADCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        Entries = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT DCC entries'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of entries recoreded by each FED'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        )
    ),
    params = cms.untracked.PSet(
        synchErrThresholdFactor = cms.untracked.double(1.0)
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than 1.0 * log10(total entries).'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ErrorsSummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT front-end status errors summary'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Counter of data towers flagged as bad in the quality summary'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        )
    )
)

process.ecalRawDataTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        BXSRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing SRP errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and SRP.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        CRC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT CRC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of CRC errors.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        BXFE = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE errors')
        ),
        BXDCCDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC-GT')
        ),
        BXFEDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing FE-DCC')
        ),
        OrbitDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-100.0)
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number DCC-GT')
        ),
        L1ASRP = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A SRP errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and SRP.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        BXTCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing TCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of bunch corssing value mismatches between DCC and TCC.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        DesyncTotal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT total FE synchronization errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        RunNumber = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT run number errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between run numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        Orbit = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT orbit number errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between LHC orbit numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        BXDCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing DCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between bunch crossing numbers recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        BXFEInvalid = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of bunch crossing value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(69.0),
                nbins = cms.untracked.int32(69),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT bunch crossing invalid value')
        ),
        DesyncByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        L1ATCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A TCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and TCC.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        FEByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of front-ends in error status in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        TrendNSyncErrors = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Accumulated trend of the number of synchronization errors (L1A & BX mismatches) between DCC and FE in this run.'),
            cumulative = cms.untracked.bool(True),
            btype = cms.untracked.string('Trend'),
            otype = cms.untracked.string('Ecal'),
            online = cms.untracked.bool(True),
            path = cms.untracked.string('Ecal/Trends/RawDataTask accumulated number of sync errors')
        ),
        EventTypePostCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing > 3490.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type post calibration BX')
        ),
        L1ADCC = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A DCC errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between L1A recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        EventTypePreCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing < 3490'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type pre calibration BX')
        ),
        EventTypeCalib = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Event type recorded in the DCC for events in bunch crossing == 3490. This plot is filled using data from the physics data stream during physics runs. It is normal to have very few entries in these cases.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(24.5),
                nbins = cms.untracked.int32(25),
                labels = cms.untracked.vstring('UNKNOWN', 
                    'COSMIC', 
                    'BEAMH4', 
                    'BEAMH2', 
                    'MTCC', 
                    'LASER_STD', 
                    'LASER_POWER_SCAN', 
                    'LASER_DELAY_SCAN', 
                    'TESTPULSE_SCAN_MEM', 
                    'TESTPULSE_MGPA', 
                    'PEDESTAL_STD', 
                    'PEDESTAL_OFFSET_SCAN', 
                    'PEDESTAL_25NS_SCAN', 
                    'LED_STD', 
                    'PHYSICS_GLOBAL', 
                    'COSMICS_GLOBAL', 
                    'HALO_GLOBAL', 
                    'LASER_GAP', 
                    'TESTPULSE_GAP', 
                    'PEDESTAL_GAP', 
                    'LED_GAP', 
                    'PHYSICS_LOCAL', 
                    'COSMICS_LOCAL', 
                    'HALO_LOCAL', 
                    'CALIB_LOCAL'),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT event type calibration BX')
        ),
        L1AFE = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Number of L1A value mismatches between DCC and FE.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(68.0),
                nbins = cms.untracked.int32(68),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('iFE')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT L1A FE errors')
        ),
        TriggerType = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT trigger type errors'),
            otype = cms.untracked.string('Ecal2P'),
            description = cms.untracked.string('Number of discrepancies between trigger type recorded in the DCC and that in CMS Event.'),
            kind = cms.untracked.string('TH1F'),
            btype = cms.untracked.string('DCC')
        ),
        FEStatus = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('FE status counter.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('ENABLED', 
                    'DISABLED', 
                    'TIMEOUT', 
                    'HEADERERROR', 
                    'CHANNELID', 
                    'LINKERROR', 
                    'BLOCKSIZE', 
                    'SUPPRESSED', 
                    'FIFOFULL', 
                    'L1ADESYNC', 
                    'BXDESYNC', 
                    'L1ABXDESYNC', 
                    'FIFOFULLL1ADESYNC', 
                    'HPARITY', 
                    'VPARITY', 
                    'FORCEDZS'),
                low = cms.untracked.double(-0.5)
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT front-end status bits %(sm)s')
        )
    )
)

process.ecalRecoSummaryTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        rechitThresholdEE = cms.untracked.double(1.2),
        rechitThresholdEB = cms.untracked.double(0.8)
    ),
    MEs = cms.untracked.PSet(
        RecoFlagReduced = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from reduced rechits.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/redRecHits_%(subdetshort)s_recoFlag')
        ),
        Chi2 = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Chi2 of the pulse reconstruction.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_Chi2')
        ),
        RecoFlagBasicCluster = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from rechits in basic clusters.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/basicClusters_recHits_%(subdetshort)s_recoFlag')
        ),
        SwissCross = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Swiss cross.'),
            otype = cms.untracked.string('EB'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_E1oE4')
        ),
        RecoFlagAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Reconstruction flags from all rechits.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                nbins = cms.untracked.int32(16),
                low = cms.untracked.double(-0.5)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_recoFlag')
        ),
        Time = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Rechit time.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-50.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_time')
        ),
        EnergyMax = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Maximum energy of the rechit.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(110),
                low = cms.untracked.double(-10.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_energyMax')
        )
    )
)

process.ecalSelectiveReadoutClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        LowIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of low interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        HighIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of high interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        ZS1Map = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1 counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy with ZS1 flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ZSFullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT ZS flagged full readout counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Number of ZS flagged but fully read out towers.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FRDroppedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT FR flagged dropped counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Number of FR flagged but dropped towers.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        MedIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of medium interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        RUForcedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT RU with forced SR counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of FORCED flag.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        ZSMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1+ZS2 counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of ZS1 and ZS2 flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FlagCounterMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower flag counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of any SR flag.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower full readout counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy with FR flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    ),
    MEs = cms.untracked.PSet(
        FR = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of full readout flag.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags%(suffix)s')
        ),
        LowInterest = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of low interest TT flags.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest TT Flags%(suffix)s')
        ),
        RUForced = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of forced selective readout.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT readout unit with SR forced%(suffix)s')
        ),
        ZS1 = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of zero suppression 1 flags.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT zero suppression 1 SR Flags%(suffix)s')
        ),
        MedInterest = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of medium interest TT flags.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT medium interest TT Flags%(suffix)s')
        ),
        HighInterest = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of high interest TT flags.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest TT Flags%(suffix)s')
        ),
        ZSReadout = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of full readout when unit is flagged as zero-suppressed.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout%(suffix)s')
        ),
        FRDropped = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Occurrence rate of unit drop when the unit is flagged as full-readout.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout%(suffix)s')
        )
    )
)

process.ecalSelectiveReadoutTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        DCCZS1stSample = cms.untracked.int32(2),
        useCondDb = cms.untracked.bool(False),
        ZSFIRWeights = cms.untracked.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 
            0.3707)
    ),
    MEs = cms.untracked.PSet(
        HighIntOutput = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Output of the ZS filter for high interest towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(60.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-60.0),
                title = cms.untracked.string('ADC counts*4')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest ZS filter output%(suffix)s')
        ),
        ZS1Map = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1 counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy with ZS1 flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FRDropped = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Number of FR flagged but dropped towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout Number%(suffix)s')
        ),
        ZSFullReadout = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Number of ZS flagged but fully read out towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout Number%(suffix)s')
        ),
        ZSFullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT ZS flagged full readout counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Number of ZS flagged but fully read out towers.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FRDroppedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT FR flagged dropped counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Number of FR flagged but dropped towers.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        LowIntOutput = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Output of the ZS filter for low interest towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(60.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-60.0),
                title = cms.untracked.string('ADC counts*4')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest ZS filter output%(suffix)s')
        ),
        LowIntPayload = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total data size from all low interest towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest payload%(suffix)s')
        ),
        TowerSize = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean data size from each readout unit.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('size (bytes)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT tower event size%(suffix)s')
        ),
        DCCSize = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Distribution of the per-DCC data size.'),
            yaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(0.0, 0.0608, 0.1216, 0.1824, 0.2432, 
                    0.304, 0.3648, 0.4256, 0.4864, 0.5472, 
                    0.608, 0.608, 1.216, 1.824, 2.432, 
                    3.04, 3.648, 4.256, 4.864, 5.472, 
                    6.08, 6.688, 7.296, 7.904, 8.512, 
                    9.12, 9.728, 10.336, 10.944, 11.552, 
                    12.16, 12.768, 13.376, 13.984, 14.592, 
                    15.2, 15.808, 16.416, 17.024, 17.632, 
                    18.24, 18.848, 19.456, 20.064, 20.672, 
                    21.28, 21.888, 22.496, 23.104, 23.712, 
                    24.32, 24.928, 25.536, 26.144, 26.752, 
                    27.36, 27.968, 28.576, 29.184, 29.792, 
                    30.4, 31.008, 31.616, 32.224, 32.832, 
                    33.44, 34.048, 34.656, 35.264, 35.872, 
                    36.48, 37.088, 37.696, 38.304, 38.912, 
                    39.52, 40.128, 40.736, 41.344),
                title = cms.untracked.string('event size (kB)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size vs DCC')
        ),
        DCCSizeProf = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Mean and spread of the per-DCC data size.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('event size (kB)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT DCC event size')
        ),
        ZSMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1+ZS2 counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of ZS1 and ZS2 flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        HighIntPayload = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total data size from all high interest towers.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest payload%(suffix)s')
        ),
        EventSize = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of per-DCC data size.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size%(suffix)s')
        ),
        FullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower full readout counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy with FR flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FlagCounterMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower flag counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of any SR flag.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        FullReadout = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Number of FR flags per event.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(200.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags Number%(suffix)s')
        ),
        RUForcedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT RU with forced SR counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of FORCED flag.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        )
    )
)

process.ecalSummaryClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        TriggerPrimitives = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s emulator error quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of emulator matching quality. A tower is red if the number of events with Et emulation error is greater than 0.1 of total events. Towers with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        DesyncByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of synchronization errors (L1A & BX mismatches) between DCC and FE in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sRawDataTask/%(prefix)sRDT FE synchronization errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        IntegrityByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of integrity errors for each FED in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sIntegrityTask/%(prefix)sIT weighted integrity errors by lumi'),
            perLumi = cms.untracked.bool(True)
        ),
        Timing = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the timing data quality. A 5x5 tower is red if the mean timing of the tower is off by more than 2.0 or RMS is greater than 6.0 (6.0 and 12.0 in forward region). Towers with total entries less than 15 are not subject to this evaluation. Since 5x5 tower timings are calculated with a tighter time-window than per-channel timings, a tower can additionally become red if its the sum of per-channel timing histogram entries is greater than per-tower histogram entries by factor 1.66666666667 (significant fraction of events fall outside the tight time-window).'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        HotCell = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sOT%(suffix)s hot cell quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the hot cell monitor. A channel is red if it has more than 100.0 times more entries than phi-ring mean in either digi, rec hit (filtered), or TP digi (filtered). Channels with less than 20 entries are not considered. Channel names of the hot cells are available in (Top) / Ecal / Errors / HotCells.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        RawData = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sSFT%(suffix)s front-end status summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the raw data (DCC and front-end) quality. A channel is red if it has nonzero events with FE status that is different from any of ENABLED, DISABLED, SUPPRESSED, FIFOFULL, FIFOFULL_L1ADESYNC, and FORCEDZS. A FED can also go red if its number of L1A desynchronization errors is greater than 1.0 * log10(total entries).'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Presample = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPOT%(suffix)s pedestal quality summary G12'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the presample data quality. A channel is red if presample mean is off by 25.0 from 200.0 or RMS is greater than 3.0. RMS threshold is 6.0 in the forward region (|eta| > 2.1). Channels with entries less than 6 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        Integrity = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sIT%(suffix)s integrity quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the data integrity. A channel is red if more than 0.01 of its entries have integrity errors.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        FEByLumi = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Total number of front-ends in error status in this lumi section.'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sStatusFlagsTask/FEStatus/%(prefix)sSFT weighted frontend errors by lumi'),
            perLumi = cms.untracked.bool(True)
        )
    ),
    params = cms.untracked.PSet(
        activeSources = cms.untracked.vstring('Integrity', 
            'RawData'),
        fedBadFraction = cms.untracked.double(0.5),
        towerBadFraction = cms.untracked.double(0.8)
    ),
    MEs = cms.untracked.PSet(
        ReportSummaryMap = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryMap'),
            otype = cms.untracked.string('Ecal'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('DCC')
        ),
        ReportSummaryContents = cms.untracked.PSet(
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string(''),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Report'),
            path = cms.untracked.string('Ecal/EventInfo/reportSummaryContents/Ecal_%(sm)s'),
            perLumi = cms.untracked.bool(True)
        ),
        NBadFEDs = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Number of FEDs with more than 50.0% of channels in bad status. Updated every lumi section.'),
            online = cms.untracked.bool(True),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0),
                nbins = cms.untracked.int32(1),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('Ecal/Errors/Number of Bad Ecal FEDs')
        ),
        ReportSummary = cms.untracked.PSet(
            kind = cms.untracked.string('REAL'),
            description = cms.untracked.string(''),
            otype = cms.untracked.string('Ecal'),
            btype = cms.untracked.string('Report'),
            path = cms.untracked.string('Ecal/EventInfo/reportSummary'),
            perLumi = cms.untracked.bool(True)
        ),
        GlobalSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Error summary used to trigger audio alarm. The value is identical to reportSummary.'),
            online = cms.untracked.bool(True),
            otype = cms.untracked.string('None'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.0),
                nbins = cms.untracked.int32(1),
                labels = cms.untracked.vstring('ECAL status'),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('Ecal/Errors/Global summary errors')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)s global summary%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the data quality. A channel is red if it is red in any one of RawData, Integrity, Timing, TriggerPrimitives, and HotCells task. A cluster of bad towers in this plot will cause the ReportSummary for the FED to go to 0 in online DQM.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        )
    )
)

process.ecalTestPulseClient = cms.untracked.PSet(
    toleranceRMS = cms.untracked.vdouble(160.0, 80.0, 10.0),
    amplitudeThreshold = cms.untracked.vdouble(1200.0, 600.0, 100.0),
    minChannelEntries = cms.untracked.int32(3),
    tolerancePNRMS = cms.untracked.vdouble(20.0, 20.0),
    sources = cms.untracked.PSet(
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Test pulse amplitude in the PN diodes.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Test pulse amplitude.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s')
        )
    ),
    MEs = cms.untracked.PSet(
        QualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT%(suffix)s test pulse quality G%(gain)s summary')
        ),
        PNQualitySummary = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Summary of test pulse data quality for PN diodes. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 12.5, 200.0 and 20.0, 20.0 for gains 1 and 16 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('MEM2P'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTPT PN test pulse quality G%(pngain)s summary')
        ),
        Quality = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Summary of test pulse data quality for crystals. A channel is red if mean amplitude is lower than the threshold, or RMS is greater than threshold. The mean and RMS thresholds are 1200.0, 600.0, 100.0 and 160.0, 80.0, 10.0 for gains 1, 6, and 12 respectively. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse quality G%(gain)s %(sm)s')
        ),
        AmplitudeRMS = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('2D distribution of the amplitude RMS. Channels with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ADC counts)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseClient/%(prefix)sTPT test pulse rms G%(gain)s')
        )
    ),
    PNAmplitudeThreshold = cms.untracked.vdouble(12.5, 200.0),
    name = cms.untracked.string('TestPulseClient')
)

process.ecalTestPulseTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        MGPAGainsPN = cms.untracked.vint32(16),
        MGPAGains = cms.untracked.vint32(12)
    ),
    MEs = cms.untracked.PSet(
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT test pulse digi occupancy%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Occupancy in test pulse events.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Shape = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Test pulse shape. One slice corresponds to one readout tower (5x5 crystals).'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(10),
                low = cms.untracked.double(0.0)
            ),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT shape %(sm)s G%(gain)s')
        ),
        PNAmplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                pngain = cms.untracked.vint32(16)
            ),
            description = cms.untracked.string('Test pulse amplitude in the PN diodes.'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('SMMEM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s')
        ),
        Amplitude = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = cms.untracked.vint32(12)
            ),
            description = cms.untracked.string('Test pulse amplitude.'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s')
        )
    )
)

process.ecalTimingClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        TimeMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 25.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(sm)s')
        ),
        TimeAllMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 7.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(7.0),
                low = cms.untracked.double(-7.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing map%(suffix)s')
        )
    ),
    params = cms.untracked.PSet(
        toleranceRMS = cms.untracked.double(6.0),
        tailPopulThreshold = cms.untracked.double(0.4),
        toleranceMean = cms.untracked.double(2.0),
        minTowerEntries = cms.untracked.int32(15),
        toleranceMeanFwd = cms.untracked.double(6.0),
        minChannelEntries = cms.untracked.int32(5),
        toleranceRMSFwd = cms.untracked.double(12.0),
        minChannelEntriesFwd = cms.untracked.int32(40),
        minTowerEntriesFwd = cms.untracked.int32(160)
    ),
    MEs = cms.untracked.PSet(
        RMSAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of per-channel timing RMS. Channels with entries less than 5 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing rms 1D summary')
        ),
        ProjEta = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of per-channel mean timing. Channels with entries less than 5 are not considered.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection eta%(suffix)s')
        ),
        FwdBkwdDiff = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Forward-backward asymmetry of per-channel mean timing. Channels with entries less than 5 are not considered.'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ - %(prefix)s-')
        ),
        FwdvBkwd = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Forward-backward correlation of per-channel mean timing. Channels with entries less than 5 are not considered.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(-25.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(prefix)s+ vs %(prefix)s-')
        ),
        MeanSM = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of per-channel timing mean. Channels with entries less than 5 are not considered.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing mean %(sm)s')
        ),
        ProjPhi = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Projection of per-channel mean timing. Channels with entries less than 5 are not considered.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing projection phi%(suffix)s')
        ),
        RMSMap = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('2D distribution of per-channel timing RMS. Channels with entries less than 5 are not considered.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rms (ns)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing rms %(sm)s')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of the timing data quality. A 5x5 tower is red if the mean timing of the tower is off by more than 2.0 or RMS is greater than 6.0 (6.0 and 12.0 in forward region). Towers with total entries less than 15 are not subject to this evaluation. Since 5x5 tower timings are calculated with a tighter time-window than per-channel timings, a tower can additionally become red if its the sum of per-channel timing histogram entries is greater than per-tower histogram entries by factor 1.66666666667 (significant fraction of events fall outside the tight time-window).'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingClient/%(prefix)sTMT timing quality %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string('Summary of the timing data quality. A channel is red if its mean timing is off by more than 2.0 or RMS is greater than 6.0. Channels with entries less than 5 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal')
        ),
        MeanAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of per-channel timing mean. Channels with entries less than 5 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTMT%(suffix)s timing mean 1D summary')
        )
    )
)

process.ecalTimingTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        energyThresholdEE = cms.untracked.double(3.0),
        energyThresholdEB = cms.untracked.double(1.0)
    ),
    MEs = cms.untracked.PSet(
        TimeMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 25.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('Crystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing %(sm)s')
        ),
        TimeAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D summary%(suffix)s')
        ),
        TimeAllMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. Hits with |t| > 7.0 ns are discarded. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(7.0),
                low = cms.untracked.double(-7.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing map%(suffix)s')
        ),
        TimeAmpAll = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-50.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(0.316227766017, 0.354813389234, 0.398107170553, 0.446683592151, 0.501187233627, 
                    0.56234132519, 0.63095734448, 0.707945784384, 0.794328234724, 0.891250938134, 
                    1.0, 1.1220184543, 1.25892541179, 1.41253754462, 1.58489319246, 
                    1.77827941004, 1.99526231497, 2.23872113857, 2.51188643151, 2.81838293126, 
                    3.16227766017, 3.54813389234, 3.98107170553, 4.46683592151, 5.01187233627, 
                    5.6234132519, 6.3095734448, 7.07945784384, 7.94328234724, 8.91250938134, 
                    10.0, 11.220184543, 12.5892541179, 14.1253754462, 15.8489319246, 
                    17.7827941004, 19.9526231497, 22.3872113857, 25.1188643151, 28.1838293126, 
                    31.6227766017, 35.4813389234, 39.8107170553, 44.6683592151, 50.1187233627, 
                    56.234132519, 63.095734448, 70.7945784384, 79.4328234724, 89.1250938134),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude summary%(suffix)s')
        ),
        TimeAmp = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Correlation between hit timing and energy. Only hits with GOOD or OUT_OF_TIME reconstruction flags are used.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(200),
                low = cms.untracked.double(-50.0),
                title = cms.untracked.string('time (ns)')
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(0.316227766017, 0.354813389234, 0.398107170553, 0.446683592151, 0.501187233627, 
                    0.56234132519, 0.63095734448, 0.707945784384, 0.794328234724, 0.891250938134, 
                    1.0, 1.1220184543, 1.25892541179, 1.41253754462, 1.58489319246, 
                    1.77827941004, 1.99526231497, 2.23872113857, 2.51188643151, 2.81838293126, 
                    3.16227766017, 3.54813389234, 3.98107170553, 4.46683592151, 5.01187233627, 
                    5.6234132519, 6.3095734448, 7.07945784384, 7.94328234724, 8.91250938134, 
                    10.0, 11.220184543, 12.5892541179, 14.1253754462, 15.8489319246, 
                    17.7827941004, 19.9526231497, 22.3872113857, 25.1188643151, 28.1838293126, 
                    31.6227766017, 35.4813389234, 39.8107170553, 44.6683592151, 50.1187233627, 
                    56.234132519, 63.095734448, 70.7945784384, 79.4328234724, 89.1250938134),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing vs amplitude %(sm)s')
        ),
        Time1D = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the mean rec hit timing. Only hits with GOOD or OUT_OF_TIME reconstruction flags and energy above threshold are used. The energy thresholds are 1.000000 and 3.000000 for EB and EE respectively.'),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                title = cms.untracked.string('time (ns)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTimingTask/%(prefix)sTMT timing 1D %(sm)s')
        )
    )
)

process.ecalTrigPrimClient = cms.untracked.PSet(
    sources = cms.untracked.PSet(
        MatchedIndex = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Counter for TP "timing" (= index withing the emulated TP whose Et matched that of the real TP)'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(6.0),
                nbins = cms.untracked.int32(6),
                labels = cms.untracked.vstring('no emul', 
                    '0', 
                    '1', 
                    '2', 
                    '3', 
                    '4'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP index')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulMatch %(sm)s')
        ),
        EtEmulError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulError %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        )
    ),
    params = cms.untracked.PSet(
        errorFractionThreshold = cms.untracked.double(0.1),
        minEntries = cms.untracked.int32(3)
    ),
    MEs = cms.untracked.PSet(
        TimingSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Emulator TP timing where the largest number of events had Et matches. Towers with entries less than 3 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP data matching emulator')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Timing summary')
        ),
        EmulQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s emulator error quality summary'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Summary of emulator matching quality. A tower is red if the number of events with Et emulation error is greater than 0.1 of total events. Towers with entries less than 3 are not considered.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        NonSingleSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Fraction of events whose emulator TP timing did not agree with the majority. Towers with entries less than 3 are not considered.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('rate')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Trigger Primitives Non Single Timing summary')
        )
    )
)

process.ecalTrigPrimTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        runOnEmul = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        LowIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower low interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of low interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        HighIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower high interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of high interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        EtMaxEmul = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the maximum Et value within one emulated TP'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/Emulated/%(prefix)sTTT Et spectrum Emulated Digis max%(suffix)s')
        ),
        EtReal = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et spectrum Real Digis%(suffix)s')
        ),
        FGEmulError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulFineGrainVetoError %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        EtVsBx = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('Mean TP Et in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('TP Et')
            ),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('1', 
                    '271', 
                    '541', 
                    '892', 
                    '1162', 
                    '1432', 
                    '1783', 
                    '2053', 
                    '2323', 
                    '2674', 
                    '2944', 
                    '3214', 
                    '3446', 
                    '3490', 
                    '3491', 
                    '3565'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et vs bx Real Digis%(suffix)s')
        ),
        EtEmulError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulError %(sm)s'),
            otype = cms.untracked.string('SM'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        MatchedIndex = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Counter for TP "timing" (= index withing the emulated TP whose Et matched that of the real TP)'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(6.0),
                nbins = cms.untracked.int32(6),
                labels = cms.untracked.vstring('no emul', 
                    '0', 
                    '1', 
                    '2', 
                    '3', 
                    '4'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP index')
            ),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT EmulMatch %(sm)s')
        ),
        EmulMaxIndex = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            description = cms.untracked.string('Distribution of the index of emulated TP with the highest Et value.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(5.5),
                nbins = cms.untracked.int32(6),
                labels = cms.untracked.vstring('no maximum', 
                    '0', 
                    '1', 
                    '2', 
                    '3', 
                    '4'),
                low = cms.untracked.double(-0.5),
                title = cms.untracked.string('TP index')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT max TP matching index%(suffix)s')
        ),
        MedIntMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower med interest counter%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string('Tower occupancy of medium interest flags.'),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        TTFlags = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            description = cms.untracked.string('Distribution of the trigger tower flags.'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(7.5),
                nbins = cms.untracked.int32(8),
                labels = cms.untracked.vstring('0', 
                    '1', 
                    '2', 
                    '3', 
                    '4', 
                    '5', 
                    '6', 
                    '7'),
                low = cms.untracked.double(-0.5),
                title = cms.untracked.string('TT flag')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('DCC'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT Flags%(suffix)s')
        ),
        TTFMismatch = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT TT flag mismatch%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            description = cms.untracked.string(''),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('TriggerTower')
        ),
        EtSummary = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('Ecal3P'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sTTT%(suffix)s Et trigger tower summary')
        ),
        EtRealMap = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile2D'),
            description = cms.untracked.string('2D distribution of the trigger primitive Et.'),
            otype = cms.untracked.string('SM'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(256.0),
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('TP Et')
            ),
            btype = cms.untracked.string('TriggerTower'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT Et map Real Digis %(sm)s')
        ),
        OccVsBx = cms.untracked.PSet(
            kind = cms.untracked.string('TProfile'),
            description = cms.untracked.string('TP occupancy in different bunch crossing intervals. This plot is filled by data from physics data stream. It is normal to have very little entries in BX >= 3490.'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(16.0),
                nbins = cms.untracked.int32(16),
                labels = cms.untracked.vstring('1', 
                    '271', 
                    '541', 
                    '892', 
                    '1162', 
                    '1432', 
                    '1783', 
                    '2053', 
                    '2323', 
                    '2674', 
                    '2944', 
                    '3214', 
                    '3446', 
                    '3490', 
                    '3491', 
                    '3565'),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('bunch crossing')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTriggerTowerTask/%(prefix)sTTT TP occupancy vs bx Real Digis%(suffix)s')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.schedule = cms.Schedule(*[ process.ecalLaserLedPath, process.ecalTestPulsePath, process.ecalPedestalPath, process.ecalMonitorPath, process.dqmEndPath, process.ecalClientPath, process.dqmOutputPath ])

### Setup source ###
process.source.runNumber = options.runNumber
process.source.runInputDir = options.runInputDir
process.source.skipFirstLumis = options.skipFirstLumis
