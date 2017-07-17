#
#  L1TReco:  Defines
#
#     L1Reco = cms.Sequence(...)
#
# which contains all L1 Reco steps needed for the current era.
#

import FWCore.ParameterSet.Config as cms

#
# These might be more widely useful...  L1T_customs?
#

def config_L1ExtraForStage2Sim(coll):
    coll.isolatedEmSource      = cms.InputTag("simCaloStage1LegacyFormatDigis","isoEm")
    coll.nonIsolatedEmSource   = cms.InputTag("simCaloStage1LegacyFormatDigis","nonIsoEm")    
    coll.forwardJetSource      = cms.InputTag("simCaloStage1LegacyFormatDigis","forJets")
    coll.centralJetSource      = cms.InputTag("simCaloStage1LegacyFormatDigis","cenJets")
    coll.tauJetSource          = cms.InputTag("simCaloStage1LegacyFormatDigis","tauJets")
    coll.isoTauJetSource       = cms.InputTag("simCaloStage1LegacyFormatDigis","isoTauJets")
    coll.etTotalSource         = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.etHadSource           = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.etMissSource          = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.htMissSource          = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.hfRingEtSumsSource    = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.hfRingBitCountsSource = cms.InputTag("simCaloStage1LegacyFormatDigis")
    coll.muonSource            = cms.InputTag("simGmtDigis")
    

from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import l1extraParticles

#
# Legacy Trigger:
#
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import l1GtRecord
from EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi import l1GtTriggerMenuLite
import EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi
conditionsInEdm = EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi.conditionDumperInEdm.clone()
import L1Trigger.GlobalTrigger.convertObjectMapRecord_cfi as _converterModule
l1L1GtObjectMap = _converterModule.convertObjectMapRecord.clone()
L1Reco_L1Extra = cms.Sequence(l1extraParticles)
L1Reco_L1Extra_L1GtRecord = cms.Sequence(l1extraParticles+l1GtRecord)
L1Reco = cms.Sequence(l1extraParticles+l1GtTriggerMenuLite+conditionsInEdm+l1L1GtObjectMap)

#
# Stage-1 Trigger
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toReplaceWith(L1Reco_L1Extra,cms.Sequence())
stage1L1Trigger.toReplaceWith(L1Reco_L1Extra_L1GtRecord,cms.Sequence())
stage1L1Trigger.toReplaceWith(L1Reco, cms.Sequence(l1extraParticles))

#
# Stage-2 Trigger:  fow now, reco Stage-1 as before:
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toReplaceWith(L1Reco_L1Extra,cms.Sequence())
stage2L1Trigger.toReplaceWith(L1Reco_L1Extra_L1GtRecord,cms.Sequence())
stage2L1Trigger.toReplaceWith(L1Reco, cms.Sequence(l1extraParticles))

#
# l1L1GtObjectMap does not work properly with fastsim
#
from Configuration.Eras.Modifier_fastSim_cff import fastSim
_L1Reco_modified = L1Reco.copyAndExclude([l1L1GtObjectMap])
fastSim.toReplaceWith(L1Reco, _L1Reco_modified)
