import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *

HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

L1TRerunHCALTP = cms.Sequence(
    simHcalTriggerPrimitiveDigis
)
