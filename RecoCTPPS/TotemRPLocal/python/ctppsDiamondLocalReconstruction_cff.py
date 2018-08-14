import FWCore.ParameterSet.Config as cms

# reco hit production
from RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits

# local track fitting
from RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi import ctppsDiamondLocalTracks

ctppsDiamondLocalReconstruction = cms.Sequence(
    ctppsDiamondRecHits *
    ctppsDiamondLocalTracks
)
