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

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtBoardMapsConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
import FWCore.Modules.printContent_cfi
process.dumpEv = FWCore.Modules.printContent_cfi.printContent.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
process.gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
process.gtDigis.DaqGtInputTag = 'source'



# shaping our Message logger to suit our needs
process.MessageLogger = cms.Service("MessageLogger",
    #suppressWarning = cms.untracked.vstring('ecalEBunpacker', 'ecalUncalibHit', 'ecalRecHit', 'ecalCosmicsHists'),
    #suppressInfo = cms.untracked.vstring('ecalEBunpacker', 'ecalUncalibHit', 'ecalRecHit', 'ecalCosmicsHists'),
    cout = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
    categories = cms.untracked.vstring('ecalDigis','ecalDccDigis','uncalibHitMaker','ecalDetIdToBeRecovered','ecalRecHit'),
    destinations = cms.untracked.vstring('cout')
)



process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(999999)
        )
process.source = cms.Source("PoolSource",
                                #debugFlag = cms.untracked.bool(True),
                                #debugVebosity = cms.untracked.uint32(10),
                                skipEvents = cms.untracked.uint32(0),
                                fileNames = cms.untracked.vstring('MYINPUTFILETYPE')
                            )

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
                                         algo = cms.string("EcalUncalibRecHitWorkerRatio")
                                         )

process.ecalDccDigis = cms.EDFilter("EcalDccDigiSkimer",
                                        EEdigiCollectionOut = cms.string('eeDigiSkim'),
                                        EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
                                        EBdigiCollectionOut = cms.string('ebDigiSkim'),
                                        EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
                                        DigiUnpacker = cms.InputTag("ecalDigis"),
                                        DigiType = cms.string('Physics')
                                    )

process.load("RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi")

process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
# make sure our calibrated rec hits can find the new name for our uncalibrated rec hits
process.ecalRecHit.EBuncalibRecHitCollection = 'uncalibHitMaker:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'uncalibHitMaker:EcalUncalibRecHitsEE'

process.timing = cms.EDFilter("EcalTimingAnalysis",
                              rootfile = cms.untracked.string('Timing_RUNNUMBER.root'),
                              CorrectBH = cms.untracked.bool(False),
                              hitProducer = cms.string('uncalibHitMaker'),
                              rhitProducer = cms.untracked.string('ecalRecHit'),
                              minNumEvt = cms.untracked.double(0),
                              FromFileName = cms.untracked.string('Emptyfile.root'),
                              TTPeakTime = cms.untracked.string('TTPeakPositionFileTiming.txt'),
                              SMAverages = cms.untracked.vdouble(5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5., 5.,
                                                                             5., 5., 5., 5.),
                              hitProducerEE = cms.string('uncalibHitMaker'),
                              rhitProducerEE = cms.untracked.string('ecalRecHit'),
                              GTRecordCollection = cms.untracked.string('gtDigis'),
                              amplThr = cms.untracked.double(15.0),
                              amplThrEE = cms.untracked.double(15.0),
                              SMCorrections = cms.untracked.vdouble(5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5., 5., 5., 5., 5.,
                                                                        5.0, 5.0, 5.0, 5.0, 5.0,
                                                                        5.0, 5.0, 5.0, 5.0),
                              BeamHaloPlus = cms.untracked.bool(True),
                              hitCollectionEE = cms.string('EcalUncalibRecHitsEE'),
                              rhitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
                              ChPeakTime = cms.untracked.string('ChPeakTimeTiming.txt'),
                              hitCollection = cms.string('EcalUncalibRecHitsEB'),
                              rhitCollection = cms.untracked.string('EcalRecHitsEB'),
                              digiProducer = cms.string('ecalDigis'),
                              CorrectEcalReadout = cms.untracked.bool(False),
                              FromFile = cms.untracked.bool(False),
                              RunStart = cms.untracked.double(1215192037.0),
                              WriteTxtFiles = cms.untracked.bool(False),
                              TimingTree = cms.untracked.bool(True),
                              AllAverage = cms.untracked.double(5.0),
                              Splash09Cor = cms.untracked.bool(False),
                              AllShift = cms.untracked.double(0.0),
                              RunLength = cms.untracked.double(2.0),
                              MinEBXtals = cms.untracked.int32(-1),
                              EBRadius = cms.untracked.double(1.4)
                              )
process.ecalDigis = process.ecalEBunpacker.clone()

process.p = cms.Path(process.gtDigis*process.ecalDigis*process.ecalDccDigis*process.uncalibHitMaker*process.ecalDetIdToBeRecovered*process.ecalRecHit*process.timing)

