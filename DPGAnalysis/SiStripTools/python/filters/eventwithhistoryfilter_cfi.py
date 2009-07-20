import FWCore.ParameterSet.Config as cms

heFilter = cms.EDFilter('EventWithHistoryEDFilter',
                        dbxRange = cms.untracked.vint32(-1,-1),
                        dbxRangeLtcyAware = cms.untracked.vint32(-1,-1),
                        absBXRange = cms.untracked.vint32(-1,-1),
                        absBXRangeLtcyAware = cms.untracked.vint32(-1,-1),
                        absBXInCycleRange = cms.untracked.vint32(-1,-1),
                        absBXInCycleRangeLtcyAware = cms.untracked.vint32(-1,-1),
                        dbxInCycleRange = cms.untracked.vint32(-1,-1),
                        dbxInCycleRangeLtcyAware = cms.untracked.vint32(-1,-1),
                        dbxTripletRange = cms.untracked.vint32(-1,-1)
                        )
