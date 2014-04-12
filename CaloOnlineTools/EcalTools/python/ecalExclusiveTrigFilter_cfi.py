import FWCore.ParameterSet.Config as cms

ecalExclusiveTrigFilter = cms.EDFilter("EcalExclusiveTrigFilter",

      # Global trigger tag
      l1GlobalReadoutRecord = cms.string("gtDigis")

)
