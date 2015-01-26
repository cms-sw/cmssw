import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.l1ExtraDQM_cfi import *

from Configuration.StandardSequences.Eras import eras # Used to modify for Run 2

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

#
# Modify for running in run 2
#
eras.run2.toModify( dqmL1ExtraParticles, etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.run2.toModify( dqmL1ExtraParticles, nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm") )
eras.run2.toModify( dqmL1ExtraParticles, etMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.run2.toModify( dqmL1ExtraParticles, htMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.run2.toModify( dqmL1ExtraParticles, forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets") )
eras.run2.toModify( dqmL1ExtraParticles, centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets") )
eras.run2.toModify( dqmL1ExtraParticles, tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets") )
eras.run2.toModify( dqmL1ExtraParticles, isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets") )
eras.run2.toModify( dqmL1ExtraParticles, isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm") )
eras.run2.toModify( dqmL1ExtraParticles, etHadSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.run2.toModify( dqmL1ExtraParticles, hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
eras.run2.toModify( dqmL1ExtraParticles, hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis") )

