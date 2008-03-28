# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms

dijetsHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2jetAve30', 'HLT2jetAve60'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


