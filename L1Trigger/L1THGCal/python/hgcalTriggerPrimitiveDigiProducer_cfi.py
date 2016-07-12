import FWCore.ParameterSet.Config as cms

fe_codec = cms.PSet( CodecName  = cms.string('HGCalBestChoiceCodec'),
                     CodecIndex = cms.uint32(1),
                     NData = cms.uint32(12),
                     DataLength = cms.uint32(8)
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
