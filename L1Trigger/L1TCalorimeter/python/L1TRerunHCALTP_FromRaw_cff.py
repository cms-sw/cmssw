import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_Data_cff import hcalDigis

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('hcalDigis'),
    cms.InputTag('hcalDigis')
)

HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

L1TRerunHCALTP_FromRAW = cms.Sequence(
    hcalDigis
    * simHcalTriggerPrimitiveDigis
)
