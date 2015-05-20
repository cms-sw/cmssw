import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.l1ExtraDQM_cfi import *
from DQM.L1TMonitor.l1ExtraDQMStage1_cfi import *

# for DQM, unpack all BxInEvent available for GCT, GMT & GT (common unpacker for GMT and GT)
# use clones dqmGctDigis and dqmGtDigis, to not interfere with RawToDigi from standard sequences

import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
dqmGctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
dqmGctDigis.inputLabel = 'rawDataCollector'
#
# unpack all five samples
dqmGctDigis.numberOfGctSamplesToUnpack = 5


import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
dqmGtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
dqmGtDigis.DaqGtInputTag = 'rawDataCollector'
#
# unpack all available BxInEvent, UnpackBxInEvent read from event setup
dqmGtDigis.UnpackBxInEvent = -1


# import the L1Extra producer, configured to run for all BX
# use a clone dqmL1ExtraParticles, to not interfere with L1Reco from standard sequences
import L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi
dqmL1ExtraParticles = L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi.l1extraParticles.clone()
#
dqmL1ExtraParticles.muonSource = 'dqmGtDigis'
dqmL1ExtraParticles.etTotalSource = 'dqmGctDigis'
dqmL1ExtraParticles.nonIsolatedEmSource = 'dqmGctDigis:nonIsoEm'
dqmL1ExtraParticles.etMissSource = 'dqmGctDigis'
dqmL1ExtraParticles.htMissSource = 'dqmGctDigis'
dqmL1ExtraParticles.forwardJetSource = 'dqmGctDigis:forJets'
dqmL1ExtraParticles.centralJetSource = 'dqmGctDigis:cenJets'
dqmL1ExtraParticles.tauJetSource = 'dqmGctDigis:tauJets'
dqmL1ExtraParticles.isolatedEmSource = 'dqmGctDigis:isoEm'
dqmL1ExtraParticles.etHadSource = 'dqmGctDigis'
dqmL1ExtraParticles.hfRingEtSumsSource = 'dqmGctDigis'
dqmL1ExtraParticles.hfRingBitCountsSource = 'dqmGctDigis'
#
dqmL1ExtraParticles.centralBxOnly = cms.bool(False)

# get stage1 digis
import L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi
dqmL1ExtraParticlesStage1 = L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi.l1extraParticles.clone()
#
dqmL1ExtraParticlesStage1.muonSource = 'dqmGtDigis'
dqmL1ExtraParticlesStage1.etTotalSource = 'caloStage1LegacyFormatDigis'
dqmL1ExtraParticlesStage1.nonIsolatedEmSource = 'caloStage1LegacyFormatDigis:nonIsoEm'
dqmL1ExtraParticlesStage1.etMissSource = 'caloStage1LegacyFormatDigis'
dqmL1ExtraParticlesStage1.htMissSource = 'caloStage1LegacyFormatDigis'
dqmL1ExtraParticlesStage1.forwardJetSource = 'caloStage1LegacyFormatDigis:forJets'
dqmL1ExtraParticlesStage1.centralJetSource = 'caloStage1LegacyFormatDigis:cenJets'
dqmL1ExtraParticlesStage1.tauJetSource = 'caloStage1LegacyFormatDigis:tauJets'
dqmL1ExtraParticlesStage1.isoTauJetSource = 'caloStage1LegacyFormatDigis:isoTauJets'
dqmL1ExtraParticlesStage1.isolatedEmSource = 'caloStage1LegacyFormatDigis:isoEm'
dqmL1ExtraParticlesStage1.etHadSource = 'caloStage1LegacyFormatDigis'
dqmL1ExtraParticlesStage1.hfRingEtSumsSource = 'caloStage1LegacyFormatDigis'
dqmL1ExtraParticlesStage1.hfRingBitCountsSource = 'caloStage1LegacyFormatDigis'
#
dqmL1ExtraParticlesStage1.centralBxOnly = cms.bool(False)

#
# Modify for running with the Stage 1 trigger. Note that these changes are already
# applied to l1extraParticles before it is cloned, but the changes are overwritten
# in the commands above. So we need to write back the correct Run 2 values.
#
from Configuration.StandardSequences.Eras import eras
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, etMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, htMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, etHadSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.stage1L1Trigger.toModify( dqmL1ExtraParticles, hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
