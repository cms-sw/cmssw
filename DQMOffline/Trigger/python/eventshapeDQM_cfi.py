import FWCore.ParameterSet.Config as cms

eventshapeDQM_Q2_top05_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top005_L1Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom005_L1Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top05_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top005_L1Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom005_L1Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top05_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top005_L1Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom05_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom005_L1Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top10_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top010_L1Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom10_cent1030 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom010_L1Centrality1030_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top10_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top010_L1Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom10_cent3050 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom010_L1Centrality3050_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_top10_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Top010_L1Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)

eventshapeDQM_Q2_bottom10_cent5070 = cms.EDAnalyzer('EventShapeDQM',
        triggerResults = cms.InputTag('TriggerResults','','HLT'),
        EPlabel = cms.InputTag("hltEvtPlaneProducer"),
        triggerPath = cms.string('HLT_Q2Bottom010_L1Centrality5070_v'),
        order = cms.int32(2),
        EPidx = cms.int32(8), #HF2
        EPlvl = cms.int32(0)
)



eventshapeDQMSequence = cms.Sequence(eventshapeDQM_Q2_top05_cent1030 * eventshapeDQM_Q2_bottom05_cent1030 * eventshapeDQM_Q2_top05_cent3050 * eventshapeDQM_Q2_bottom05_cent3050 * eventshapeDQM_Q2_top05_cent5070 * eventshapeDQM_Q2_bottom05_cent5070 * eventshapeDQM_Q2_top10_cent1030 * eventshapeDQM_Q2_bottom10_cent1030 * eventshapeDQM_Q2_top10_cent3050 * eventshapeDQM_Q2_bottom10_cent3050 * eventshapeDQM_Q2_top10_cent5070 * eventshapeDQM_Q2_bottom10_cent5070)
