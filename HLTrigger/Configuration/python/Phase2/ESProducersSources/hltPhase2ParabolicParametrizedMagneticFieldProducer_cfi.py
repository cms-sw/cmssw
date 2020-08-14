import FWCore.ParameterSet.Config as cms

from MagneticField.ParametrizedEngine.autoParabolicParametrizedField_cfi import (
    ParametrizedMagneticFieldProducer as _ParametrizedMagneticFieldProducer,
)

hltPhase2ParabolicParametrizedMagneticFieldProducer = _ParametrizedMagneticFieldProducer.clone(
    label="ParabolicMf"
)
