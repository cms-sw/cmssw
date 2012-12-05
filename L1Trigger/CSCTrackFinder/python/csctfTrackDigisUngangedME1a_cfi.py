import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi import csctfTrackDigis as csctfTrackDigisUngangedME1a

csctfTrackDigisUngangedME1a.SectorProcessor.PTLUT.PtMethod = cms.untracked.uint32(33) # no triple ganging in ME11a
csctfTrackDigisUngangedME1a.SectorProcessor.gangedME1a = cms.untracked.bool(False)
csctfTrackDigisUngangedME1a.SectorProcessor.firmwareSP = cms.uint32(20120730) #core 20120730
csctfTrackDigisUngangedME1a.SectorProcessor.initializeFromPSet = cms.bool(True) 
