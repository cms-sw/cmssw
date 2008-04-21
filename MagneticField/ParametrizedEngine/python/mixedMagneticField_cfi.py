import FWCore.ParameterSet.Config as cms

from MagneticField.Engine.volumeBasedMagneticField_cfi import *
MixedMagneticFieldProducer = cms.ESProducer("MixedMagneticFieldProducer",
    parametrizationLabel = cms.untracked.string('parametrizedFieldMap'),
    fullMapScale = cms.double(0.936863), ## 3.81113/4.06797

    label = cms.untracked.string(''),
    fullMapLabel = cms.untracked.string('fullFieldMap')
)

VolumeBasedMagneticFieldESProducer.label = 'fullFieldMap'
ParametrizedMagneticFieldProducer.label = 'parametrizedFieldMap'
ParametrizedMagneticFieldProducer.version = 'OAE_1103l_071212'
ParametrizedMagneticFieldProducer.parameters = cms.PSet(
    BValue = cms.string('3_8T')
)

