import FWCore.ParameterSet.Config as cms

TauSpinnerZHFilter = cms.EDFilter("TauSpinnerFilter",
                                  ntaus = cms.int32(2)
                                  )

TauSpinnerWHpmFilter = cms.EDFilter("TauSpinnerFilter",
                                    ntaus = cms.int32(1)
                                    )

