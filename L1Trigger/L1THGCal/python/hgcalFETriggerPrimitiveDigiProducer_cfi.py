import FWCore.ParameterSet.Config as cms

hgcalFETriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalFETriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    TriggerGeometry = cms.PSet(
        TriggerGeometryName = cms.string('NullGeometry'),
        eeSDName = cms.string('HGCalEESensitive'),
        fhSDName = cms.string('HGCalHESiliconSensitive'),
        bhSDName = cms.string('HGCalHEScintillatorSensitive'),
        )
    )
