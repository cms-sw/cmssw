# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

dijetsHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_DiJetAve15','HLT_DiJetAve30','HLT_DiJetAve50','HLT_Jet30','HLT_Jet50'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


