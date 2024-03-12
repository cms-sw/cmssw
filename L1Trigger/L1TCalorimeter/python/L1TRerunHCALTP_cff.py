import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *

HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

L1TRerunHCALTP = cms.Sequence(
    simHcalTriggerPrimitiveDigis
)
# foo bar baz
# j2MZoGH0IqyPR
# WQWLaDamwXYTq
