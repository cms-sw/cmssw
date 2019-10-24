import FWCore.ParameterSet.Config as cms

MEtoEDMConverter = cms.EDProducer("MEtoEDMConverter",
    Name = cms.untracked.string('MEtoEDMConverter'),
    Verbosity = cms.untracked.int32(0), # 0 provides no output
                                        # 1 provides basic output
                                        # 2 provide more detailed output
    Frequency = cms.untracked.int32(50),
    MEPathToSave = cms.untracked.string(''),
)

