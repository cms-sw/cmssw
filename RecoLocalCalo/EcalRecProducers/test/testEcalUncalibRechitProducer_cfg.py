import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO', eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.CUDACore.ProcessAcceleratorCUDA_cfi')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
#process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
#process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')






# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_dataRun2_HLT_v2', '')


process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(100)
    input = cms.untracked.int32(1000)
)

# load data using the DAQ source
import sys, os, inspect
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
process.load('sourceFromRawCmggpu_cff')

#-----------------------------------------
# CMSSW/Hcal non-DQM Related Module import
#-----------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
#process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")
process.load("RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi")

# load both cpu and gpu plugins
#
# ../cfipython/slc7_amd64_gcc700/RecoLocalCalo/EcalRecProducers/ecalUncalibRecHitProducerGPU_cfi.py
#
process.load("RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitProducerGPU_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi")

# for validation of gpu multifit products
process.load("RecoLocalCalo.EcalRecProducers.ecalCPUUncalibRecHitProducer_cfi")
#
# ../cfipython/slc7_amd64_gcc700/RecoLocalCalo/EcalRecProducers/ecalCPUUncalibRecHitProducer_cfi.py
#

process.load("EventFilter.EcalRawToDigi.ecalRawToDigiGPU_cfi")
process.load("EventFilter.EcalRawToDigi.ecalElectronicsMappingGPUESProducer_cfi")

#process.ecalUncalibRecHitProducerGPU.kernelsVersion = 0
#process.ecalUncalibRecHitProducerGPU.kernelMinimizeThreads = cms.vuint32(16, 1, 1)
#
# process.ecalUncalibRecHitProducerGPU.shouldRunTimingComputation = cms.bool(False)
#


process.load("RecoLocalCalo.EcalRecProducers.ecalPedestalsGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalGainRatiosGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalPulseShapesGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalPulseCovariancesGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalSamplesCorrelationGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalTimeBiasCorrectionsGPUESProducer_cfi")
process.load("RecoLocalCalo.EcalRecProducers.ecalTimeCalibConstantsGPUESProducer_cfi")

#process.ecalMultiFitUncalibRecHitgpu.algoPSet.threads = cms.vint32(256, 1, 1)


process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitParametersGPUESProducer_cfi")



##
## force HLT configuration for ecalMultiFitUncalibRecHit
##

process.ecalMultiFitUncalibRecHit.algoPSet = cms.PSet( 
      EBtimeFitLimits_Upper = cms.double( 1.4 ),
      EEtimeFitLimits_Lower = cms.double( 0.2 ),
      timealgo = cms.string( "None" ),   # ----> no timing computation for CPU version
      EBtimeNconst = cms.double( 28.5 ),
      prefitMaxChiSqEE = cms.double( 10.0 ),
      outOfTimeThresholdGain12mEB = cms.double( 5.0 ),
      outOfTimeThresholdGain12mEE = cms.double( 1000.0 ),
      EEtimeFitParameters = cms.vdouble( -2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277 ),
      prefitMaxChiSqEB = cms.double( 25.0 ),
      simplifiedNoiseModelForGainSwitch = cms.bool( True ),
      EBtimeFitParameters = cms.vdouble( -2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621 ),
      selectiveBadSampleCriteriaEB = cms.bool( False ),
      dynamicPedestalsEB = cms.bool( False ),
      useLumiInfoRunHeader = cms.bool( False ),
      EBamplitudeFitParameters = cms.vdouble( 1.138, 1.652 ),
      doPrefitEE = cms.bool( False ),
      dynamicPedestalsEE = cms.bool( False ),
      selectiveBadSampleCriteriaEE = cms.bool( False ),
      outOfTimeThresholdGain61pEE = cms.double( 1000.0 ),
      outOfTimeThresholdGain61pEB = cms.double( 5.0 ),
      activeBXs = cms.vint32( -5, -4, -3, -2, -1, 0, 1, 2, 3, 4 ),
      doPrefitEB = cms.bool( False ),
      addPedestalUncertaintyEE = cms.double( 0.0 ),
      addPedestalUncertaintyEB = cms.double( 0.0 ),
      gainSwitchUseMaxSampleEB = cms.bool( True ),
      EEtimeNconst = cms.double( 31.8 ),
      EEamplitudeFitParameters = cms.vdouble( 1.89, 1.4 ),
      outOfTimeThresholdGain12pEB = cms.double( 5.0 ),
      gainSwitchUseMaxSampleEE = cms.bool( False ),
      mitigateBadSamplesEB = cms.bool( False ),
      outOfTimeThresholdGain12pEE = cms.double( 1000.0 ),
      ampErrorCalculation = cms.bool( False ),
      mitigateBadSamplesEE = cms.bool( False ),
      amplitudeThresholdEB = cms.double( 10.0 ),
      amplitudeThresholdEE = cms.double( 10.0 ),
      EBtimeFitLimits_Lower = cms.double( 0.2 ),
      EEtimeFitLimits_Upper = cms.double( 1.4 ),
      outOfTimeThresholdGain61mEE = cms.double( 1000.0 ),
      EEtimeConstantTerm = cms.double( 1.0 ),
      EBtimeConstantTerm = cms.double( 0.6 ),
      outOfTimeThresholdGain61mEB = cms.double( 5.0 )
)     
      
##    
    
    
    
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.ecalRecHit

    
    
#process.load("RecoLocalCalo.EcalRecProducers.ecalRecHitGPU_cfi")
#process.ecalRecHitGPU
    


#process.hcalDigis.silent = cms.untracked.bool(False)
#process.hcalDigis.InputLabel = rawTag
process.ecalDigis = process.ecalEBunpacker.clone()
process.ecalDigis.InputLabel = cms.InputTag('rawDataCollector')
#process.hbheprerecogpu.processQIE11 = cms.bool(True)

process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("test_uncalib.root")
)

#process.out = cms.OutputModule("AsciiOutputModule",
#    outputCommands = cms.untracked.vstring(
#        'keep *_ecalMultiFitUncalibRecHit_*_*', 
#    ),
#    verbosity = cms.untracked.uint32(0)
#)
process.finalize = cms.EndPath(process.out)

process.bunchSpacing = cms.Path(
    process.bunchSpacingProducer
)

process.digiPath = cms.Path(
    #process.hcalDigis
    process.ecalDigis
    *process.ecalRawToDigiGPU    
)

process.recoPath = cms.Path(
    #(process.ecalMultiFitUncalibRecHit+process.ecalDetIdToBeRecovered)
    process.ecalMultiFitUncalibRecHit
    #*process.ecalRecHit
#   gpu
    *process.ecalUncalibRecHitProducerGPU
    *process.ecalCPUUncalibRecHitProducer
    #*process.ecalRecHitGPU
)

process.schedule = cms.Schedule(
    process.bunchSpacing,
    process.digiPath,
    process.recoPath,
#    process.ecalecalLocalRecoSequence
    process.finalize
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    TryToContinue = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)
