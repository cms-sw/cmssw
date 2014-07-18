import FWCore.ParameterSet.Config as cms

TauSpinnerZHFilter = cms.EDFilter("TauSpinnerFilter",
                                  src = cms.InputTag('TauSpinnerGen','TauSpinnerWT'), 
                                  ntaus = cms.int32(2)
                                  )

TauSpinnerWHpmFilter = cms.EDFilter("TauSpinnerFilter",
                                    src = cms.InputTag('TauSpinnerGen','TauSpinnerWT'),
                                    ntaus = cms.int32(1)
                                    )

