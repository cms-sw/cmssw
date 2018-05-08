import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *

from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
hgcl1tpg_step = cms.Sequence(hgcalTriggerPrimitives)

from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff import *
EcalEBtp_step = cms.Sequence(simEcalEBTriggerPrimitiveDigis)

#from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cff import *
#HcalTPsimulation_step = cms.Sequence(hcalTTPSequence)

from Configuration.StandardSequences.SimL1Emulator_cff import *
L1simulation_step = cms.Sequence(SimL1Emulator)

from L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff import *
L1TrackTrigger_step = cms.Sequence(L1TrackletTracks)

reprocess_L1Phase2_MC = cms.Sequence(
    #hgcl1tpg_step +
    #EcalEBtp_step + 
    L1TrackTrigger_step +
    #HcalTPsimulation_step
    L1simulation_step
)
