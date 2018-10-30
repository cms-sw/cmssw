import FWCore.ParameterSet.Config as cms

CalibrationTrackSelectorFromDetIdList = cms.EDProducer("CalibrationTrackSelectorFromDetIdList",
                                                       Input = cms.InputTag("generalTracks"),
                                                       verbose = cms.untracked.bool(False),
                                                       selections = cms.VPSet(cms.PSet(selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),    # TEC minus
                                                                              cms.PSet(selection=cms.untracked.vstring("0x1e0c0000-0x1c080000")),    # TEC plus
                                                                              cms.PSet(selection=cms.untracked.vstring("0x1e000000-0x1a000000")),    # TOB
                                                                              cms.PSet(selection=cms.untracked.vstring("0x1e000000-0x16000000")),    # TIB
                                                                              cms.PSet(selection=cms.untracked.vstring("0x1e006000-0x18002000")),    # TID minus
                                                                              cms.PSet(selection=cms.untracked.vstring("0x1e006000-0x18004000")),    # TID plus
                                                                              )
                                                       )
