import FWCore.ParameterSet.Config as cms
##-------- Jet Triggers --------------------
HLTL1Jet6U = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_L1Jet6U')
)

HLTDiJetAve15U8E29 = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_DiJetAve15U_8E29')
)

HLTDiJetAve30U8E29 = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_DiJetAve30U_8E29')
)
##-------- Zero Bias Trigger ---------------
HLTZeroBias = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_ZeroBias')
)
##-------- Photon Triggers -----------------
HLTPhoton15L1R = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_Photon15_L1R')
)

HLTPhoton15TrackIsoL1R = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_Photon15_TrackIso_L1R')
)

HLTPhoton15LooseEcalIsoL1R = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_Photon15_LooseEcalIso_L1R')
)

HLTPhoton20L1R = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_Photon20_L1R')
)

HLTPhoton30L1R8E29 = cms.EDFilter("TriggerFilter",
    triggerResultsTag  = cms.InputTag('TriggerResults','','HLT'),
    triggerProcessName = cms.string('HLT'),
    DEBUG              = cms.bool(False),
    triggerName        = cms.string('HLT_Photon30_L1R_8E29')
)
HLTPhoton = cms.Sequence(HLTPhoton15L1R + HLTPhoton20L1R + HLTPhoton30L1R8E29 + HLTPhoton15TrackIsoL1R + HLTPhoton15LooseEcalIsoL1R)

##-------- Electrons Triggers --------------

##-------- Muon Triggers -------------------
