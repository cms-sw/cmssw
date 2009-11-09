import FWCore.ParameterSet.Config as cms

hltLogMonitorFilter = cms.EDFilter("HLTLogMonitorFilter",
    default_threshold = cms.uint32(10),
    categories = cms.VPSet(
        cms.PSet(
            name = cms.string('Category'),
            threshold = cms.uint32(20)
        ),
        cms.PSet(
            name = cms.string('Unprescaled'),
            threshold = cms.uint32(1)
        ),
        cms.PSet(
            name = cms.string('Disabled'),
            threshold = cms.uint32(0)
        )
    )
)
