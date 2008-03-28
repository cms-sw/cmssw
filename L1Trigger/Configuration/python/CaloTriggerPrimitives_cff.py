import FWCore.ParameterSet.Config as cms

from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
# ECAL TPG
from Geometry.EcalMapping.EcalMapping_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_cff import *
# HCAL TPG
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
CaloTriggerPrimitives = cms.Sequence(ecalTriggerPrimitiveDigis*hcalTriggerPrimitiveDigis)
ecalTriggerPrimitiveDigis.Label = 'ecalDigis'
hcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'

