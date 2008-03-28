# The following comments couldn't be translated into the new config version:

# provide list of HLT paths you want
import FWCore.ParameterSet.Config as cms

hltHighLevel = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring(),
    andOr = cms.bool(True), ## true: OR of those on your list 

    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)


