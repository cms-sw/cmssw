import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from SimCalorimetry.Configuration.SimCalorimetry_cff import *

from SimGeneral.DataMixingModule.mixOne_data_on_data_cfi import *

# Run after the DataMixer only.
#
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# 
#
# clone these sequences:

DM_EcalTriggerPrimitiveDigis = simEcalTriggerPrimitiveDigis.clone()
DM_EcalDigis = simEcalDigis.clone()

# Re-define inputs to point at DataMixer output
DM_EcalTriggerPrimitiveDigis.Label = cms.string('mixData')
DM_EcalTriggerPrimitiveDigis.InstanceEB = cms.string('EBDigiCollectionDM')
DM_EcalTriggerPrimitiveDigis.InstanceEE = cms.string('EEDigiCollectionDM')
#
DM_EcalDigis.digiProducer = cms.string('mixData')
DM_EcalDigis.EBdigiCollection = cms.string('EBDigiCollectionDM')
DM_EcalDigis.EEdigiCollection = cms.string('EEDigiCollectionDM')

ecalDigiSequenceDM = cms.Sequence(DM_EcalTriggerPrimitiveDigis*DM_EcalDigis)

# same for Hcal:

# clone these sequences:

DM_HcalTriggerPrimitiveDigis = simHcalTriggerPrimitiveDigis.clone()
DM_HcalDigis = simHcalDigis.clone()

# Re-define inputs to point at DataMixer output
DM_HcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('mixData'),cms.InputTag('mixData'))
DM_HcalDigis.digiLabel = cms.InputTag("mixData")

hcalDigiSequenceDM = cms.Sequence(DM_HcalTriggerPrimitiveDigis+DM_HcalDigis)

postDM_Digi = cms.Sequence(ecalDigiSequenceDM+hcalDigiSequenceDM)

pdatamix = cms.Sequence(mixData+postDM_Digi)

