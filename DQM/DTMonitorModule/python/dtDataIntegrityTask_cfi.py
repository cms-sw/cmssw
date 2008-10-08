import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.Service("DTDataIntegrityTask",
                                  getSCInfo = cms.untracked.bool(True),
                                  hltMode = cms.untracked.bool(False)
)


