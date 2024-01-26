import FWCore.ParameterSet.Config as cms

def customizeHLTforAlpakaEcalLocalReco(process):
    process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
    if hasattr(process, 'hltEcalDigisGPU'):
        process.hltEcalDigisPortable = cms.EDProducer("EcalRawToDigiPortable@alpaka",
            FEDs = process.hltEcalDigisGPU.FEDs,
            InputLabel = process.hltEcalDigisGPU.InputLabel,
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
            ),
            digisLabelEB = process.hltEcalDigisGPU.digisLabelEB,
            digisLabelEE = process.hltEcalDigisGPU.digisLabelEE,
            maxChannelsEB = process.hltEcalDigisGPU.maxChannelsEB,
            maxChannelsEE = process.hltEcalDigisGPU.maxChannelsEE,
            mightGet = cms.optional.untracked.vstring
        )
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.hltEcalDigisPortable)

        process.load("EventFilter.EcalRawToDigi.ecalElectronicsMappingHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalElectronicsMappingHostESProducer)

        delattr(process, 'hltEcalDigisGPU')
        delattr(process, 'ecalElectronicsMappingGPUESProducer')

    if hasattr(process, 'hltEcalDigisFromGPU'):
        process.hltEcalDigisFromGPU = cms.EDProducer( "EcalDigisFromPortableProducer",
            digisInLabelEB = cms.InputTag( 'hltEcalDigisPortable','ebDigis' ),
            digisInLabelEE = cms.InputTag( 'hltEcalDigisPortable','eeDigis' ),
            digisOutLabelEB = cms.string( "ebDigis" ),
            digisOutLabelEE = cms.string( "eeDigis" ),
            produceDummyIntegrityCollections = cms.bool( False )
        )

    if hasattr(process, 'hltEcalUncalibRecHitGPU'):
        process.hltEcalUncalibRecHitPortable = cms.EDProducer("EcalUncalibRecHitProducerPortable@alpaka",
            EBtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EBtimeConstantTerm,
            EBtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Lower,
            EBtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EBtimeFitLimits_Upper,
            EBtimeNconst = process.hltEcalUncalibRecHitGPU.EBtimeNconst,
            EEtimeConstantTerm = process.hltEcalUncalibRecHitGPU.EEtimeConstantTerm,
            EEtimeFitLimits_Lower = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Lower,
            EEtimeFitLimits_Upper = process.hltEcalUncalibRecHitGPU.EEtimeFitLimits_Upper,
            EEtimeNconst = process.hltEcalUncalibRecHitGPU.EEtimeNconst,
            alpaka = cms.untracked.PSet(
                backend = cms.untracked.string('')
            ),
            amplitudeThresholdEB = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEB,
            amplitudeThresholdEE = process.hltEcalUncalibRecHitGPU.amplitudeThresholdEE,
            digisLabelEB = cms.InputTag("hltEcalDigisPortable","ebDigis"),
            digisLabelEE = cms.InputTag("hltEcalDigisPortable","eeDigis"),
            kernelMinimizeThreads = process.hltEcalUncalibRecHitGPU.kernelMinimizeThreads,
            mightGet = cms.optional.untracked.vstring,
            outOfTimeThresholdGain12mEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12mEB,
            outOfTimeThresholdGain12mEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12mEE,
            outOfTimeThresholdGain12pEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12pEB,
            outOfTimeThresholdGain12pEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain12pEE,
            outOfTimeThresholdGain61mEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61mEB,
            outOfTimeThresholdGain61mEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61mEE,
            outOfTimeThresholdGain61pEB = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61pEB,
            outOfTimeThresholdGain61pEE = process.hltEcalUncalibRecHitGPU.outOfTimeThresholdGain61pEE,
            recHitsLabelEB = process.hltEcalUncalibRecHitGPU.recHitsLabelEB,
            recHitsLabelEE = process.hltEcalUncalibRecHitGPU.recHitsLabelEE,
            shouldRunTimingComputation = process.hltEcalUncalibRecHitGPU.shouldRunTimingComputation
        )
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.hltEcalUncalibRecHitPortable)

        process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitConditionsHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalMultifitConditionsHostESProducer)

        process.ecalMultifitParametersSource = cms.ESSource("EmptyESSource",
            firstValid = cms.vuint32(1),
            iovIsRunNotTime = cms.bool(True),
            recordName = cms.string('EcalMultifitParametersRcd')
        )
        process.load("RecoLocalCalo.EcalRecProducers.ecalMultifitParametersHostESProducer_cfi")
        process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask.add(process.ecalMultifitParametersHostESProducer)

        delattr(process, 'hltEcalUncalibRecHitGPU')

        if hasattr(process, 'hltEcalUncalibRecHitFromSoA'):
            process.hltEcalUncalibRecHitFromSoA = cms.EDProducer("EcalUncalibRecHitSoAToLegacy",
                isPhase2 = process.hltEcalUncalibRecHitFromSoA.isPhase2,
                mightGet = cms.optional.untracked.vstring,
                recHitsLabelCPUEB = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEB,
                recHitsLabelCPUEE = process.hltEcalUncalibRecHitFromSoA.recHitsLabelCPUEE,
                uncalibRecHitsPortableEB = cms.InputTag("hltEcalUncalibRecHitPortable","EcalUncalibRecHitsEB"),
                uncalibRecHitsPortableEE = cms.InputTag("hltEcalUncalibRecHitPortable","EcalUncalibRecHitsEE")
            )

        if hasattr(process, 'hltEcalUncalibRecHitSoA'):
            delattr(process, 'hltEcalUncalibRecHitSoA')

    process.HLTDoFullUnpackingEgammaEcalTask = cms.ConditionalTask(process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerTask, process.HLTPreshowerTask)

    return process

def customizeHLTforAlpaka(process):
    process = customizeHLTforAlpakaEcalLocalReco(process)

    return process

