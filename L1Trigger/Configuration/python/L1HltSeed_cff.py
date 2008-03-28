import FWCore.ParameterSet.Config as cms

# the sequence to be run in EvF before running any HLT paths
#
# Jim Brooke, Vasile Ghete
#
# NB - contact HLT team when this file is edited
#      so changes can be mirrored in ConfDB
# Setup
from L1Trigger.Configuration.L1HltSeedSetup_cff import *
import copy
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
# Modules
gtDigis = copy.deepcopy(l1GtUnpack)
import copy
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
gctDigis = copy.deepcopy(l1GctHwDigis)
import copy
from L1Trigger.GlobalTrigger.gtDigis_cfi import *
l1GtObjectMap = copy.deepcopy(gtDigis)
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
# the sequence
L1HltSeed = cms.Sequence(gtDigis*gctDigis*l1GtObjectMap*l1extraParticles)
gtDigis.DaqGtInputTag = 'rawDataCollector'
gtDigis.ActiveBoardsMask = 0x0101
gtDigis.UnpackBxInEvent = 1
gctDigis.inputLabel = 'rawDataCollector'
l1GtObjectMap.GmtInputTag = 'gtDigis'
l1GtObjectMap.GctInputTag = 'gctDigis'
l1GtObjectMap.ProduceL1GtDaqRecord = False
l1GtObjectMap.ProduceL1GtEvmRecord = False
l1GtObjectMap.WritePsbL1GtDaqRecord = False
l1GtParameters.TotalBxInEvent = 1
l1extraParticles.muonSource = 'gtDigis'
l1extraParticles.isolatedEmSource = cms.InputTag("gctDigis","isoEm")
l1extraParticles.nonIsolatedEmSource = cms.InputTag("gctDigis","nonIsoEm")
l1extraParticles.forwardJetSource = cms.InputTag("gctDigis","forJets")
l1extraParticles.centralJetSource = cms.InputTag("gctDigis","cenJets")
l1extraParticles.tauJetSource = cms.InputTag("gctDigis","tauJets")
l1extraParticles.etTotalSource = 'gctDigis'
l1extraParticles.etHadSource = 'gctDigis'
l1extraParticles.etMissSource = 'gctDigis'

