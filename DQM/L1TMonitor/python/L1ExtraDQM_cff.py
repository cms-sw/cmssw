import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.l1ExtraDQM_cfi import *
from DQM.L1TMonitor.l1ExtraDQMStage1_cfi import *

# for DQM, unpack all BxInEvent available for GCT, GMT & GT (common unpacker for GMT and GT)
# use clones dqmGctDigis and dqmGtDigis, to not interfere with RawToDigi from standard sequences

import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
dqmGctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone(
  inputLabel = 'rawDataCollector'
)
#
# unpack all five samples
dqmGctDigis.numberOfGctSamplesToUnpack = 5


import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
dqmGtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone(
  DaqGtInputTag = 'rawDataCollector'
)
#
# unpack all available BxInEvent, UnpackBxInEvent read from event setup
dqmGtDigis.UnpackBxInEvent = -1


# import the L1Extra producer, configured to run for all BX
# use a clone dqmL1ExtraParticles, to not interfere with L1Reco from standard sequences
import L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi
dqmL1ExtraParticles = L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi.l1extraParticles.clone(
  #
  muonSource = 'dqmGtDigis',
  etTotalSource = 'dqmGctDigis',
  nonIsolatedEmSource = 'dqmGctDigis:nonIsoEm',
  etMissSource = 'dqmGctDigis',
  htMissSource = 'dqmGctDigis',
  forwardJetSource = 'dqmGctDigis:forJets',
  centralJetSource = 'dqmGctDigis:cenJets',
  tauJetSource = 'dqmGctDigis:tauJets',
  isolatedEmSource = 'dqmGctDigis:isoEm',
  etHadSource = 'dqmGctDigis',
  hfRingEtSumsSource = 'dqmGctDigis',
  hfRingBitCountsSource = 'dqmGctDigis',
  centralBxOnly = False
)
# get stage1 digis
import L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi
dqmL1ExtraParticlesStage1 = L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi.l1extraParticles.clone(
  #
  muonSource = 'dqmGtDigis',
  etTotalSource = 'caloStage1LegacyFormatDigis',
  nonIsolatedEmSource = 'caloStage1LegacyFormatDigis:nonIsoEm',
  etMissSource = 'caloStage1LegacyFormatDigis',
  htMissSource = 'caloStage1LegacyFormatDigis',
  forwardJetSource = 'caloStage1LegacyFormatDigis:forJets',
  centralJetSource = 'caloStage1LegacyFormatDigis:cenJets',
  tauJetSource = 'caloStage1LegacyFormatDigis:tauJets',
  isoTauJetSource = 'caloStage1LegacyFormatDigis:isoTauJets',
  isolatedEmSource = 'caloStage1LegacyFormatDigis:isoEm',
  etHadSource = 'caloStage1LegacyFormatDigis',
  hfRingEtSumsSource = 'caloStage1LegacyFormatDigis',
  hfRingBitCountsSource = 'caloStage1LegacyFormatDigis',
  centralBxOnly = False
)
#
# Modify for running with the Stage 1 trigger. Note that these changes are already
# applied to l1extraParticles before it is cloned, but the changes are overwritten
# in the commands above. So we need to write back the correct Run 2 values.
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( dqmL1ExtraParticles, etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, etMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, htMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, etHadSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticles, hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
