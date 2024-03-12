import FWCore.ParameterSet.Config as cms

occupancyplots = cms.EDAnalyzer('OccupancyPlots',
                                multiplicityMaps = cms.VInputTag(cms.InputTag("ssclusmultprod")),
                                occupancyMaps = cms.VInputTag(cms.InputTag("ssclusoccuprod")),
                                wantedSubDets = cms.VPSet()
                                      )

# foo bar baz
# foJjFe0JR5QGW
# rh1TM7uLSSx0c
