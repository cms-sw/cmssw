import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from SimCalorimetry.Configuration.SimCalorimetry_cff import *
from SimMuon.Configuration.SimMuon_cff import *

from SimGeneral.PreMixingModule.mixOne_premix_on_sim_cfi import *

# Run after the DataMixer only.
#
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# 
#
# clone these sequences:

DMEcalTriggerPrimitiveDigis = simEcalTriggerPrimitiveDigis.clone()
DMEcalEBTriggerPrimitiveDigis = simEcalEBTriggerPrimitiveDigis.clone()
DMEcalDigis = simEcalDigis.clone()
DMEcalPreshowerDigis = simEcalPreshowerDigis.clone()

# Re-define inputs to point at DataMixer output
DMEcalTriggerPrimitiveDigis.Label = cms.string('mixData')
DMEcalTriggerPrimitiveDigis.InstanceEB = cms.string('')
DMEcalTriggerPrimitiveDigis.InstanceEE = cms.string('')
#
DMEcalEBTriggerPrimitiveDigis.barrelEcalDigis = 'mixData'
#
DMEcalDigis.digiProducer = cms.string('mixData')
DMEcalDigis.EBdigiCollection = cms.string('')
DMEcalDigis.EEdigiCollection = cms.string('')
DMEcalDigis.trigPrimProducer = cms.string('DMEcalTriggerPrimitiveDigis')
#
DMEcalPreshowerDigis.digiProducer = cms.string('mixData')
#DMEcalPreshowerDigis.ESdigiCollection = cms.string('ESDigiCollectionDM')

ecalDigiTaskDM = cms.Task(DMEcalTriggerPrimitiveDigis, DMEcalDigis, DMEcalPreshowerDigis)
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
_phase2_ecalDigiTaskDM = ecalDigiTaskDM.copy()
_phase2_ecalDigiTaskDM.add(DMEcalEBTriggerPrimitiveDigis)
phase2_common.toReplaceWith(ecalDigiTaskDM, _phase2_ecalDigiTaskDM)

# same for Hcal:

# clone these sequences:

DMHcalTriggerPrimitiveDigis = simHcalTriggerPrimitiveDigis.clone()
DMHcalDigis = simHcalDigis.clone()
DMHcalTTPDigis = simHcalTTPDigis.clone()

# Re-define inputs to point at DataMixer output
DMHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('mixData'),cms.InputTag('mixData'))
DMHcalTriggerPrimitiveDigis.inputUpgradeLabel = cms.VInputTag(cms.InputTag('mixData:HBHEQIE11DigiCollection'),cms.InputTag('mixData:HFQIE10DigiCollection'))
DMHcalDigis.digiLabel = cms.string('mixData')
DMHcalTTPDigis.HFDigiCollection = cms.InputTag("mixData")

hcalDigiTaskDM = cms.Task(DMHcalTriggerPrimitiveDigis, DMHcalDigis, DMHcalTTPDigis)

postDMDigi = cms.Task(ecalDigiTaskDM, hcalDigiTaskDM, muonDigiTask)

# disable adding noise to HCAL cells with no MC signal
#mixData.doEmpty = False

#
# TrackingParticle Producer is now part of the mixing module, so
# it is no longer run here.
#
from SimGeneral.PileupInformation.AddPileupSummary_cfi import *


pdatamixTask = cms.Task(mixData, postDMDigi, addPileupInfo)
pdatamix = cms.Sequence(pdatamixTask)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
def _fastSimDigis(process):
    # pretend these digis have been through digi2raw and raw2digi, by using the approprate aliases
    # use an alias to make the mixed track collection available under the usual label
    from FastSimulation.Configuration.DigiAliases_cff import loadDigiAliases
    loadDigiAliases(process, premix=True)
modifyDataMixerPreMix_fastSimDigis = fastSim.makeProcessModifier(_fastSimDigis)
