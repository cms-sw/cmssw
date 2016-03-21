#
#  L1TReco:  Defines
#
#     L1Reco = cms.Sequence(...)
#
# which contains all L1 Reco steps needed for the current era.
#

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#
# These might be more widely useful...  L1T_customs?
#


def config_L1ExtraForStage1Raw(coll):
    coll.isolatedEmSource      = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
    coll.nonIsolatedEmSource   = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")    
    coll.forwardJetSource      = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
    coll.centralJetSource      = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
    coll.tauJetSource          = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
    coll.isoTauJetSource       = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets")
    coll.etTotalSource         = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.etHadSource           = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.etMissSource          = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.htMissSource          = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.hfRingEtSumsSource    = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis")
    coll.muonSource            = cms.InputTag("gtDigis")
    
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
    

#
# Legacy Trigger:
#
if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
    print "L1TReco Sequence configured for Run1 (Legacy) trigger. "
    from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
    l1extraParticles.centralBxOnly = False
    from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *
    from EventFilter.L1GlobalTriggerRawToDigi.l1GtTriggerMenuLite_cfi import *
    import EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi
    conditionsInEdm = EventFilter.L1GlobalTriggerRawToDigi.conditionDumperInEdm_cfi.conditionDumperInEdm.clone()
    from L1Trigger.GlobalTrigger.convertObjectMapRecord_cfi import *
    l1L1GtObjectMap = convertObjectMapRecord.clone()
    L1Reco_L1Extra = cms.Sequence(l1extraParticles)
    L1Reco_L1Extra_L1GtRecord = cms.Sequence(l1extraParticles+l1GtRecord)
    L1Reco = cms.Sequence(l1extraParticles+l1GtTriggerMenuLite+conditionsInEdm+l1L1GtObjectMap)


#
# Stage-1 Trigger
#
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TReco Sequence configured for Stage-1 (2015) trigger. "    
    from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
    config_L1ExtraForStage1Raw(l1extraParticles)
    L1Reco = cms.Sequence(l1extraParticles)

#
# Stage-2 Trigger:  fow now, reco Stage-1 as before:
#
if eras.stage2L1Trigger.isChosen():
    print "L1TReco Sequence configured for Stage-2 (2016) trigger. "    
    from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
    config_L1ExtraForStage1Raw(l1extraParticles)
    L1Reco = cms.Sequence(l1extraParticles)

if eras.fastSim.isChosen():
    # fastsim runs L1Reco and HLT in one step
    # this requires to set :
    from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
    l1extraParticles.centralBxOnly = True
