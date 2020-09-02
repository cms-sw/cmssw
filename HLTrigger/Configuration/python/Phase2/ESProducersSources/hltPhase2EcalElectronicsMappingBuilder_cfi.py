import FWCore.ParameterSet.Config as cms

from Geometry.EcalMapping.EcalMapping_cfi import (
    EcalElectronicsMappingBuilder as _EcalElectronicsMappingBuilder,
)

hltPhase2EcalElectronicsMappingBuilder = _EcalElectronicsMappingBuilder.clone()
