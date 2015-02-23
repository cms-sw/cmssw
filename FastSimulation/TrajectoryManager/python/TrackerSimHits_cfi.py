import FWCore.ParameterSet.Config as cms

TrackerSimHitsBlock = cms.PSet(
    TrackerSimHits = cms.PSet(
        # Smallest charged particle pT for which SimHit's are saved (GeV/c)
        pTmin = cms.untracked.double(0.2),
        # Save SimHit's only for the first loop
        firstLoop = cms.untracked.bool(True)
    )
)

#
# Modify for running in Run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify( TrackerSimHitsBlock.TrackerSimHits, pTmin = 0.1 )
