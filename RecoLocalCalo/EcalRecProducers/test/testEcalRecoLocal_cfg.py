import FWCore.ParameterSet.Config as cms
process = cms.Process("testEcalRecoLocal")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# get timing service up for profiling
process.TimerService = cms.Service("TimerService")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


# unpacking
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

# load trivial conditions
process.EcalTrivialConditionRetriever = cms.ESSource("EcalTrivialConditionRetriever",
                                                         adcToGeVEBConstant = cms.untracked.double(0.035),
                                                         adcToGeVEEConstant = cms.untracked.double(0.06),
                                                         pedWeights = cms.untracked.vdouble(0.333, 0.333, 0.333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                         amplWeights = cms.untracked.vdouble(-0.333, -0.333, -0.333, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                                                         jittWeights = cms.untracked.vdouble(0.041, 0.041, 0.041, 0.0, 1.325, -0.05, -0.504, -0.502, -0.390, 0.0)
                                                     )


# get uncalibrechits with weights method
import RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi
process.ecalUncalibHitWeights = RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi.ecalWeightUncalibRecHit.clone()
process.ecalUncalibHitWeights.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHitWeights.EEdigiCollection = 'ecalEBunpacker:eeDigis'

# get uncalibrechits with fit method
import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHitFixedAlphaBetaFit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()
process.ecalUncalibHitFixedAlphaBetaFit.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHitFixedAlphaBetaFit.EEdigiCollection = 'ecalEBunpacker:eeDigis'

# get uncalibrechits with ratio method
import RecoLocalCalo.EcalRecProducers.ecalRatioUncalibRecHit_cfi
process.ecalUncalibHitRatio = RecoLocalCalo.EcalRecProducers.ecalRatioUncalibRecHit_cfi.ecalRatioUncalibRecHit.clone()
process.ecalUncalibHitRatio.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHitRatio.EEdigiCollection = 'ecalEBunpacker:eeDigis'


# get uncalibrechits with ratio method
import RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi
process.ecalUncalibHitGlobal = RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi.ecalGlobalUncalibRecHit.clone()
process.ecalUncalibHitGlobal.EBdigiCollection = 'ecalEBunpacker:ebDigis'
process.ecalUncalibHitGlobal.EEdigiCollection = 'ecalEBunpacker:eeDigis'


# get rechits e.g. from the weights
process.load("CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi")
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalUncalibHit:EcalUncalibRecHitsEE'


process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32(10) )
process.source = cms.Source("PoolSource",
            fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/v1/000/067/838/006945C8-40A5-DD11-BD7E-001617DBD556.root')
            #fileNames = cms.untracked.vstring('file:/data/franzoni/data/Commissioning08_RAW_v1_67838-006945C8-40A5-DD11-BD7E-001617DBD556.root')
            #fileNames = cms.untracked.vstring('file:/data/franzoni/data/h4b-sm6/h4b.00016428.A.0.0.root')
                )


process.outputmodule = cms.OutputModule("PoolOutputModule",
                                          outputCommands = cms.untracked.vstring(
            'drop *',
            'keep *_ecalUncalibHit*_*_*',
            'keep *_ecalRecHit_*_*'
          ),
  fileName = cms.untracked.string('testEcalLocalRecoA.root')
)

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.ecalTestRecoLocal = cms.Sequence(process.ecalEBunpacker
                                         *process.ecalUncalibHitWeights
                                         *process.ecalUncalibHitFixedAlphaBetaFit
                                         *process.ecalUncalibHitRatio
                                         *process.ecalUncalibHitGlobal
                                         *process.ecalRecHit
                                         *process.outputmodule
                                         #*process.dumpEv
                                        )

process.p = cms.Path(process.ecalTestRecoLocal)
