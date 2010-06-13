import FWCore.ParameterSet.Config as cms

# ECAL TPG
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_with_suppressed_cfi import *

# HCAL TPG
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *

#
CaloTriggerPrimitives = cms.Sequence(simEcalTriggerPrimitiveDigis*simHcalTriggerPrimitiveDigis)

