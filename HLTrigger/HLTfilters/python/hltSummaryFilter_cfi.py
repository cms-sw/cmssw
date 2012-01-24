import FWCore.ParameterSet.Config as cms

hltSummaryFilter = cms.EDFilter("HLTSummaryFilter",
    summary = cms.InputTag("hltTriggerSummaryAOD","","HLT"), # trigger summary
    member  = cms.InputTag("hlt1jet30","","HLT"),      # filter or collection
    saveTags = cms.bool( False ),
    cut     = cms.string("pt>80"),                     # cut on trigger object
    minN    = cms.int32(1)                  # min. # of passing objects needed
)
