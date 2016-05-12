import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam

fe_codec = cms.PSet( CodecName  = cms.string('HGCalBestChoiceCodec'),
                     CodecIndex = cms.uint32(1),
                     NData = cms.uint32(12),
                     DataLength = cms.uint32(8),
                     linLSB = cms.double(100./1024.),
                     #take the following parameters from the digitization config file
                     adcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC,
                     adcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits,
                     tdcsaturation = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC,
                     tdcnBits = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits,
                     tdcOnsetfC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
                   )


cluster_algo =  cms.PSet( AlgorithmName = cms.string('FullModuleSumAlgo'),
                                 FECodec = fe_codec )

hgcalTriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalTriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    TriggerGeometry = cms.PSet(
        TriggerGeometryName = cms.string('HGCalTriggerGeometryImp1'),
        L1TCellsMapping = cms.FileInPath("L1Trigger/L1THGCal/data/cellsToTriggerCellsMap.txt"),
        eeSDName = cms.string('HGCalEESensitive'),
        fhSDName = cms.string('HGCalHESiliconSensitive'),
        bhSDName = cms.string('HGCalHEScintillatorSensitive'),
        ),
    FECodec = fe_codec,
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )
