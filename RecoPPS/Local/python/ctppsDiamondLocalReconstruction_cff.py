import FWCore.ParameterSet.Config as cms

# Temporary customization of GT - to be removed once the tag is included in the GTs
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import GlobalTag
GlobalTag.toGet.append(
  cms.PSet(record = cms.string('PPSTimingCalibrationLUTRcd'),
           tag = cms.string('PPSDiamondTimingCalibrationLUT_test')
  )
)

# reco hit production
from RecoPPS.Local.ctppsDiamondRecHits_cfi import ctppsDiamondRecHits

# local track fitting
from RecoPPS.Local.ctppsDiamondLocalTracks_cfi import ctppsDiamondLocalTracks

ctppsDiamondLocalReconstructionTask = cms.Task(
    ctppsDiamondRecHits,
    ctppsDiamondLocalTracks
)
ctppsDiamondLocalReconstruction = cms.Sequence(ctppsDiamondLocalReconstructionTask)
