import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 

fe_codec = cms.PSet( CodecName  = cms.string('HGCalTriggerCellBestChoiceCodec'),
                     CodecIndex = cms.uint32(2),
                     NData = cms.uint32(12),
                     MaxCellsInModule = cms.uint32(116),
                     DataLength = cms.uint32(16),
                     linLSB = cms.double(100./1024.),
                     triggerCellTruncationBits = cms.uint32(0),
                     #take the following parameters from the digitization config file
                     adcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
                     adcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits,
                     tdcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
                     tdcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits,
                     tdcOnsetfC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
                     )

cluster_algo =  cms.PSet( AlgorithmName = cms.string('FullModuleSumAlgo'),
                          FECodec = fe_codec.clone(),

                          calib_parameters = cms.PSet( cellLSB =  cms.double( fe_codec.linLSB.value() * (2 ** fe_codec.triggerCellTruncationBits.value() ) ),
                                                       fCperMIPee = recoparam.HGCalUncalibRecHit.HGCEEConfig.fCPerMIP,
                                                       fCperMIPfh = recoparam.HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP,
                                                       dEdXweights = recocalibparam.HGCalRecHit.layerWeights,
                                                       thickCorr = recocalibparam.HGCalRecHit.thicknessCorrection                     
                                                       )
                          )

hgcalTriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalTriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    #bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )

hgcalTriggerPrimitiveDigiFEReproducer = cms.EDProducer(
    "HGCalTriggerDigiFEReproducer",
    feDigis = cms.InputTag('hgcalTriggerPrimitiveDigiProducer'),
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )
