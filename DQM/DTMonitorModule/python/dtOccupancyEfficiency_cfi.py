import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dtOccupancyMonitor = DQMEDAnalyzer('DTOccupancyEfficiency',
    # switch for verbosity
    debug = cms.untracked.bool(False),
    # label for dtDigis
    digiLabel = cms.string('muonDTDigis'),
    # labels of 4D and 1D hits
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits'),
)



