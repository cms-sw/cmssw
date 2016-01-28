import FWCore.ParameterSet.Config as cms

# Start with Standard Digitization:

from SimCalorimetry.Configuration.SimCalorimetry_cff import *

from SimGeneral.DataMixingModule.mixOne_simraw_on_sim_cfi import *

# Run after the DataMixer only.
#
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# 
#
# clone these sequences:

DMEcalTriggerPrimitiveDigis = simEcalTriggerPrimitiveDigis.clone()
DMEcalDigis = simEcalDigis.clone()
DMEcalPreshowerDigis = simEcalPreshowerDigis.clone()

# Re-define inputs to point at DataMixer output
DMEcalTriggerPrimitiveDigis.Label = cms.string('mixData')
DMEcalTriggerPrimitiveDigis.InstanceEB = cms.string('')
DMEcalTriggerPrimitiveDigis.InstanceEE = cms.string('')
#
DMEcalDigis.digiProducer = cms.string('mixData')
DMEcalDigis.EBdigiCollection = cms.string('')
DMEcalDigis.EEdigiCollection = cms.string('')
DMEcalDigis.trigPrimProducer = cms.string('DMEcalTriggerPrimitiveDigis')
#
DMEcalPreshowerDigis.digiProducer = cms.string('mixData')
#DMEcalPreshowerDigis.ESdigiCollection = cms.string('ESDigiCollectionDM')

ecalDigiSequenceDM = cms.Sequence(DMEcalTriggerPrimitiveDigis*DMEcalDigis*DMEcalPreshowerDigis)

# same for Hcal:

# clone these sequences:

DMHcalTriggerPrimitiveDigis = simHcalTriggerPrimitiveDigis.clone()
DMHcalDigis = simHcalDigis.clone()
DMHcalTTPDigis = simHcalTTPDigis.clone()

# Re-define inputs to point at DataMixer output
DMHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('mixData'),cms.InputTag('mixData'))
DMHcalDigis.digiLabel = cms.string('mixData')
DMHcalTTPDigis.HFDigiCollection = cms.InputTag("mixData")

hcalDigiSequenceDM = cms.Sequence(DMHcalTriggerPrimitiveDigis+DMHcalDigis*DMHcalTTPDigis)

postDMDigi = cms.Sequence(ecalDigiSequenceDM+hcalDigiSequenceDM)

# disable adding noise to HCAL cells with no MC signal
#mixData.doEmpty = False

#
# TrackingParticle Producer is now part of the mixing module, so
# it is no longer run here.
#
from SimGeneral.PileupInformation.AddPileupSummaryPreMixed_cfi import *



pdatamix = cms.Sequence(mixData+postDMDigi+addPileupInfo)

from Configuration.StandardSequences.Eras import eras
if eras.fastSim.isChosen():
    # pretend these digis have been through digi2raw and raw2digi, by using the approprate aliases
    # use an alias to make the mixed track collection available under the usual label
    from FastSimulation.Configuration.DigiAliases_cff import loadDigiAliases
    loadDigiAliases(premix = True)
    from FastSimulation.Configuration.DigiAliases_cff import generalTracks,ecalPreshowerDigis,ecalDigis,hcalDigis,muonDTDigis,muonCSCDigis,muonRPCDigis
