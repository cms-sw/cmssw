import FWCore.ParameterSet.Config as cms

TrackerOfflineValidationSummary = cms.EDFilter("TrackerOfflineValidationSummary",
   moduleDirectoryInOutput = cms.string("AlCaReco/TkAl"),  # has to be the same as in TrackerOfflineValidation_Dqm_cff
   useFit                  = cms.bool(False),
)
