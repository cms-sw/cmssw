import FWCore.ParameterSet.Config as cms

# EventBasicData Module default configuration
modEventBasicData = cms.EDFilter("EventBasicData",
    # Output ROOT file name
    oOFileName = cms.untracked.string('EventBasicData_out.root')
)


