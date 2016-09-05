import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

fe_codec = cms.PSet( CodecName  = cms.string('HGCalBestChoiceCodec'),
                     CodecIndex = cms.uint32(1),
                     NData = cms.uint32(12),
                     DataLength = cms.uint32(8),
                     linLSB = cms.double(100./1024.),
                     triggerCellTruncationBits = cms.uint32(2),
                     #take the following parameters from the digitization config file
                     adcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
                     adcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits,
                     tdcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
                     tdcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits,
                     tdcOnsetfC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
                   )

geometry = cms.PSet( TriggerGeometryName = cms.string('HGCalTriggerGeometryHexImp2'),
                     L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/triggercell_mapping.txt"),
                     L1TModulesMapping = cms.FileInPath("L1Trigger/L1THGCal/data/module_mapping.txt"),
                     eeSDName = cms.string('HGCalEESensitive'),
                     fhSDName = cms.string('HGCalHESiliconSensitive'),
                     bhSDName = cms.string('HGCalHEScintillatorSensitive'),
                   )


cluster_algo =  cms.PSet( AlgorithmName = cms.string('FullModuleSumAlgo'),
                                 FECodec = fe_codec )

hgcalTriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalTriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    #bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    TriggerGeometry = geometry,
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )

hgcalTriggerPrimitiveDigiFEReproducer = cms.EDProducer(
    "HGCalTriggerDigiFEReproducer",
    feDigis = cms.InputTag('hgcalTriggerPrimitiveDigiProducer'),
    TriggerGeometry = geometry,
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )
