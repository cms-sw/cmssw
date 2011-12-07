import FWCore.ParameterSet.Config as cms

DTDataIntegrityTask = cms.Service("DTDataIntegrityTask",
                                  getSCInfo = cms.untracked.bool(True),
                                  fedIntegrityFolder = cms.untracked.string("DT/FEDIntegrity"),
                                  processingMode     = cms.untracked.string("Online")
)


