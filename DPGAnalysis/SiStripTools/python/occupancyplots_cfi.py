import FWCore.ParameterSet.Config as cms

occupancyplots = cms.EDAnalyzer('OccupancyPlots',
                                multiplicityMaps = cms.VInputTag(cms.InputTag("ssclusmultprod")),
                                occupancyMaps = cms.VInputTag(cms.InputTag("ssclusoccuprod")),
                                wantedSubDets = cms.VPSet()
                                      )

