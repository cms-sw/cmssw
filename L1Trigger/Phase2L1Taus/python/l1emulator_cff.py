import FWCore.ParameterSet.Config as cms

l1emulator = cms.Sequence()

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *

from L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
l1emulator += L1THGCalTriggerPrimitives

from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff import *
l1emulator += simEcalEBTriggerPrimitiveDigis

from L1Trigger.TrackFindingTracklet.l1tTTTracksFromTrackletEmulation_cfi import *
L1TRK_NAME  = "l1tTTTracksFromTrackletEmulation"
L1TRK_LABEL = "Level1TTTracks"

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
l1emulator += offlineBeamSpot

l1emulator += l1tTTTracksFromTrackletEmulation

from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag( cms.InputTag(L1TRK_NAME, L1TRK_LABEL) )
l1emulator += TrackTriggerAssociatorTracks

from L1Trigger.VertexFinder.l1tVertexProducer_cfi import *
l1emulator += l1tVertexProducer

from Configuration.StandardSequences.SimL1Emulator_cff import *
l1emulator += SimL1Emulator

from L1Trigger.Phase2L1ParticleFlow.l1tPFTracksFromL1Tracks_cfi import *
l1emulator += l1tPFTracksFromL1Tracks

from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *
l1emulator += l1tParticleFlow

from L1Trigger.L1CaloTrigger.Phase1L1TJets_cff import *
l1emulator += L1TPhase1JetsSequence

