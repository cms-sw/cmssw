import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.l1ExtraDQM_cfi import *


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
