import FWCore.ParameterSet.Config as cms

process = cms.Process("ECALTIMING")
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.GlobalTag.globaltag = 'GR10_P_V4::All'


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(999999)
        )
process.source = cms.Source("PoolSource",
                                #debugFlag = cms.untracked.bool(True),
                                #debugVebosity = cms.untracked.uint32(10),
#                                skipEvents = cms.untracked.uint32(1200000),
                                skipEvents = cms.untracked.uint32(0),
                                fileNames = cms.untracked.vstring('MYINPUTFILETYPE')
                            )

process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

#process.src1 = cms.ESSource("EcalTrivialConditionRetriever",
#                                jittWeights = cms.untracked.vdouble(0.04, 0.04, 0.04, 0.0, 1.32,
#                                                                            -0.05, -0.5, -0.5, -0.4, 0.0),
#                                pedWeights = cms.untracked.vdouble(0.333, 0.333, 0.333, 0.0, 0.0,
#                                                                           0.0, 0.0, 0.0, 0.0, 0.0),
#                                amplWeights = cms.untracked.vdouble(-0.333, -0.333, -0.333, 0.0, 0.0,
#                                                                            1.0, 0.0, 0.0, 0.0, 0.0)
#                            )


process.uncalibHitMaker = cms.EDProducer("EcalUncalibRecHitProducer",
                                         EBdigiCollection = cms.InputTag("ecalDccDigis","ebDigiSkim"),
                                         EEdigiCollection = cms.InputTag("ecalDccDigis","eeDigiSkim"),
                                         EBhitCollection = cms.string("EcalUncalibRecHitsEB"),
                                         EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
                                         EBtimeFitParameters = cms.vdouble(-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01),
                                         EEtimeFitParameters = cms.vdouble(-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01),
                                         EBamplitudeFitParameters = cms.vdouble(1.138,1.652),
                                         EEamplitudeFitParameters = cms.vdouble(1.890,1.400),
                                         EBtimeFitLimits_Lower = cms.double(0.2),
                                         EBtimeFitLimits_Upper = cms.double(1.4),
                                         EEtimeFitLimits_Lower = cms.double(0.2),
                                         EEtimeFitLimits_Upper = cms.double(1.4),

                                         #outOfTimeThreshold = cms.double(0.25),
                                         #amplitudeThresholdEB = cms.double(20 * 1),
                                         #amplitudeThresholdEE = cms.double(20 * 1),

                                         #ebPulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
                                         #eePulseShape = cms.vdouble( 5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194 ),
                                         algo = cms.string("EcalUncalibRecHitWorkerRatio")
                                                                                  )

#
#process.uncalibHitMaker = cms.EDProducer("EcalUncalibRecHitProducer",
#                                             EEdigiCollection = cms.InputTag("ecalDccDigis","eeDigiSkim"),
#                                             betaEE = cms.double(1.37),
#                                             alphaEE = cms.double(1.63),
#                                             EBdigiCollection = cms.InputTag("ecalDccDigis","ebDigiSkim"),
#                                             EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
#                                             AlphaBetaFilename = cms.untracked.string('NOFILE'),
#                                             betaEB = cms.double(1.7),
#                                             MinAmplEndcap = cms.double(14.0),
#                                             MinAmplBarrel = cms.double(8.0),
#                                             alphaEB = cms.double(1.2),
#                                             UseDynamicPedestal = cms.bool(True),
#                                             EBhitCollection = cms.string('EcalUncalibRecHitsEB'),
#                                             algo = cms.string("EcalUncalibRecHitWorkerFixedAlphaBetaFit")
#                                         )

process.ecalDccDigis = cms.EDFilter("EcalDccDigiSkimer",
                                        EEdigiCollectionOut = cms.string('eeDigiSkim'),
                                        EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
                                        EBdigiCollectionOut = cms.string('ebDigiSkim'),
                                        EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
                                        DigiUnpacker = cms.InputTag("ecalEBunpacker"),
                                        DigiType = cms.string('Laser')
                                    )

process.timing = cms.EDFilter("EcalTimingAnalysis",
                                  rootfile = cms.untracked.string('Laser_RUNNUMBER.root'),
                                  CorrectBH = cms.untracked.bool(False),
                                  hitProducer = cms.string('uncalibHitMaker'),
                                  minNumEvt = cms.untracked.double(20.0),
                                  FromFileName = cms.untracked.string('Emptyfile.root'),
                                  TTPeakTime = cms.untracked.string('TTPeakPositionFileLaser.txt'),
                                  SMAverages = cms.untracked.vdouble(5.0703, 5.2278, 5.2355, 5.1548, 5.1586,
                                                                             5.1912, 5.1576, 5.1625, 5.1269, 5.643,
                                                                             5.6891, 5.588, 5.5978, 5.6508, 5.6363,
                                                                             5.5972, 5.6784, 5.6108, 5.6866, 5.6523,
                                                                             5.6666, 5.7454, 5.729, 5.7751, 5.7546,
                                                                             5.7835, 5.7529, 5.5691, 5.6677, 5.5662,
                                                                             5.6308, 5.7097, 5.6773, 5.76, 5.8025,
                                                                             5.9171, 5.8771, 5.8926, 5.9011, 5.8447,
                                                                             5.8142, 5.8475, 5.7123, 5.6216, 5.6713,
                                                                             5.3747, 5.3564, 5.39, 5.8081, 5.8081,
                                                                             5.1818, 5.1125, 5.1334, 5.2581),
                                  hitProducerEE = cms.string('uncalibHitMaker'),
                                  GTRecordCollection = cms.untracked.string('NO'),
                                  amplThr = cms.untracked.double(25.0),
                                  SMCorrections = cms.untracked.vdouble(0.0, 0.0, 0.0, 0.0, 0.0,
                                                                                0.0, 0.0, 0.0, 0.0, 0.2411,
                                                                                0.2411, 0.2221, 0.2221, -0.1899, -0.1899,
                                                                                -0.1509, -0.1509, 0.0451, 0.0451, -0.1709,
                                                                                -0.1709, 0.2221, 0.2221, -0.1899, -0.1899,
                                                                                -0.1359, -0.1359, -0.1359, -0.1359, 0.2221,
                                                                                0.2221, -0.2099, -0.2099, 0.2531, 0.2531,
                                                                                -0.1359, -0.1359, -0.2099, -0.2099, 0.2411,
                                                                                0.2411, 0.2531, 0.2531, -0.1709, -0.1709,
                                                                                0.0, 0.0, 0.0, 0.0, 0.0,
                                                                                0.0, 0.0, 0.0, 0.0),
                                  BeamHaloPlus = cms.untracked.bool(True),
                                  hitCollectionEE = cms.string('EcalUncalibRecHitsEE'),
                                  ChPeakTime = cms.untracked.string('ChPeakTimeLaser.txt'),
                                  hitCollection = cms.string('EcalUncalibRecHitsEB'),
                                  digiProducer = cms.string('ecalEBunpacker'),
                                  CorrectEcalReadout = cms.untracked.bool(False),
                                  FromFile = cms.untracked.bool(False),
                                  RunStart = cms.untracked.double(1215192037.0),
                                  RunLength = cms.untracked.double(2.0),
                                  EBRadius = cms.untracked.double(1.4)
                              )

process.p = cms.Path(process.ecalEBunpacker*process.ecalDccDigis*process.uncalibHitMaker*process.timing)

