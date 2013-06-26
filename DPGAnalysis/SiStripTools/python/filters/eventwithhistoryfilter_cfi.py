import FWCore.ParameterSet.Config as cms

heFilter = cms.EDFilter('EventWithHistoryEDFilter',
                        filterConfigurations = cms.untracked.VPSet(
                           cms.PSet(
                                    dbxRange = cms.untracked.vint32(-1,-1),
                                    dbxRangeLtcyAware = cms.untracked.vint32(-1,-1),
                                    absBXRange = cms.untracked.vint32(-1,-1),
                                    absBXRangeLtcyAware = cms.untracked.vint32(-1,-1),
                                    absBXInCycleRange = cms.untracked.vint32(-1,-1),
                                    absBXInCycleRangeLtcyAware = cms.untracked.vint32(-1,-1),
                                    dbxInCycleRange = cms.untracked.vint32(-1,-1),
                                    dbxInCycleRangeLtcyAware = cms.untracked.vint32(-1,-1),
                                    dbxTripletRange = cms.untracked.vint32(-1,-1),
                                    dbxGenericRange = cms.untracked.vint32(-1,-1),
                                    dbxGenericFirst = cms.untracked.uint32(0),dbxGenericLast = cms.untracked.uint32(1)
                                    )
                           )
                        )
