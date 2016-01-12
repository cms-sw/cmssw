import FWCore.ParameterSet.Config as cms

occupancyplots = cms.EDAnalyzer('OccupancyPlots',
                                multiplicityMaps = cms.VInputTag(cms.InputTag("ssclusmultprod")),
                                occupancyMaps = cms.VInputTag(cms.InputTag("ssclusoccuprod")),
                                checkWithLabels = cms.bool(False),
                                wantedSubDets = cms.VPSet()
                                      )

