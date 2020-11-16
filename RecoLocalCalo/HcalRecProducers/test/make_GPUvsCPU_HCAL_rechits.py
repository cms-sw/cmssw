import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
#from Configuration.ProcessModifiers.gpu_cff import gpu

process = cms.Process('RECOgpu', eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('HeterogeneousCore.CUDAServices.CUDAService_cfi')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

#-----------------------------------------
# INPUT
#-----------------------------------------

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('/store/data/Run2018D/EphemeralHLTPhysics1/RAW/v1/000/323/775/00000/A27DFA33-8FCB-BE42-A2D2-1A396EEE2B6E.root')
)

process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)

process.input = cms.Path( process.hltGetRaw )

#-----------------------------------------
# CMSSW/Hcal non-DQM Related Module import
#-----------------------------------------

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
process.load("EventFilter.HcalRawToDigi.HcalRawToDigi_cfi")
process.load("RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi")

process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")

#-----------------------------------------
# CMSSW/Hcal GPU related files
#-----------------------------------------

process.load("RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPUTask_cff")
process.load("RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi")
process.hcalCPURecHitsProducer.recHitsM0LabelIn = cms.InputTag("hbheRecHitProducerGPU","")
process.hcalCPURecHitsProducer.recHitsM0LabelOut = cms.string("")

#-----------------------------------------
# Temporary customization (things not implemented on the GPU)
#-----------------------------------------

## the one below is taken directly from the DB, regard M0
#process.hbheprereco.algorithm.correctForPhaseContainment = cms.bool(False)

## do always 8 pulse
process.hbheprereco.algorithm.chiSqSwitch = cms.double(-1)

## to match hard coded setting (will be fixed on CPU)
process.hbheprereco.algorithm.nMaxItersMin = cms.int32(50)

#-----------------------------------------
# Final Custmization for Run3
#-----------------------------------------

# we will not run arrival Time at HLT
process.hbheprereco.algorithm.calculateArrivalTime = cms.bool(False)

## we do not need this
process.hbheprereco.algorithm.applyLegacyHBMCorrection = cms.bool(False)

# we only run Mahi at HLT
process.hbheprereco.algorithm.useM3 = cms.bool(False)

# we will not have the HPD noise flags in Run3, as will be all siPM
process.hbheprereco.setLegacyFlagsQIE8 = cms.bool(False)
process.hbheprereco.setNegativeFlagsQIE8 = cms.bool(False)
process.hbheprereco.setNoiseFlagsQIE8 = cms.bool(False)
process.hbheprereco.setPulseShapeFlagsQIE8 = cms.bool(False)

# for testing M0 only
##process.hbheprereco.algorithm.useMahi = cms.bool(False)

#-----------------------------------------
# OUTPUT
#-----------------------------------------

#process.out = cms.OutputModule("AsciiOutputModule",
#    outputCommands = cms.untracked.vstring(
#        'keep *_*_*_*', 
#    ),
#    verbosity = cms.untracked.uint32(0)
#)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("GPUvsCPU_HCAL_rechits.root")
)

#---------------

process.finalize = cms.EndPath(process.out)

process.bunchSpacing = cms.Path(
    process.bunchSpacingProducer
)

#-----------------------------------------
# gpu test
#-----------------------------------------

process.digiPathCPU = cms.Path(
    process.hcalDigis 
)

process.recoPathCPU = cms.Path(
     process.hbheprereco
)

#---------------

## hcalCPUDigisProducer <-- this convert the GPU digi on cpu (for dqm)
process.recoPathGPU = cms.Path(
    process.hbheRecHitProducerGPUSequence
    * process.hcalCPURecHitsProducer
)

#---------------

process.schedule = cms.Schedule(
    process.input,
    process.digiPathCPU,
    process.recoPathCPU,
    process.recoPathGPU,
    process.finalize
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    SkipEvent = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)

# report CUDAService messages
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.categories.append("CUDAService")
