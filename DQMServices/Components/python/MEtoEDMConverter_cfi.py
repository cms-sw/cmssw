import FWCore.ParameterSet.Config as cms

MEtoEDMConverter = cms.EDFilter("MEtoEDMConverter",
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    Name = cms.untracked.string('MEtoEDMConverter')
)




