import FWCore.ParameterSet.Config as cms

from Geometry.MTDNumberingBuilder.mtdTopology_cfi import mtdTopology as _mtdTopology

hltPhase2mtdTopology = _mtdTopology.clone()
