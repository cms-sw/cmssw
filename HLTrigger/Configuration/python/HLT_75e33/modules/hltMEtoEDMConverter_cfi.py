import FWCore.ParameterSet.Config as cms

hltMEtoEDMConverter = cms.EDProducer('MEtoEDMConverter',
                                  Name = cms.untracked.string('MEtoEDMConverter'),
                                  Verbosity = cms.untracked.int32(0),
                                  Frequency = cms.untracked.int32(50),
                                  MEPathToSave = cms.untracked.string('')
                                  )
