import FWCore.ParameterSet.Config as cms

eventshapeDQM_Q2_top05_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Top005_Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Bottom005_Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top05_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Top005_Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Bottom005_Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top05_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Top005_Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hiEvtPlane"),
        triggerPath = cms.string('HLT_HIQ2Bottom005_Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)



eventshapeDQMSequence = cms.Sequence(eventshapeDQM_Q2_top05_cent1030 * eventshapeDQM_Q2_bottom05_cent1030 * eventshapeDQM_Q2_top05_cent3050 * eventshapeDQM_Q2_bottom05_cent3050 * eventshapeDQM_Q2_top05_cent5070 * eventshapeDQM_Q2_bottom05_cent5070 )
