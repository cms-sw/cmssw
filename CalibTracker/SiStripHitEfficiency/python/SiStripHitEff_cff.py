import FWCore.ParameterSet.Config as cms

anEff = cms.EDAnalyzer("HitEff",
                       Debug = cms.bool(False),
                       Layer = cms.int32(0), # =0 means do all layers
                       #combinatorialTracks = cms.InputTag("ctfWithMaterialTracksP5"),
                       #combinatorialTracks = cms.InputTag("TrackRefitterP5"),
                       #combinatorialTracks = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                       combinatorialTracks = cms.InputTag("generalTracks"),
                       #trajectories = cms.InputTag("ctfWithMaterialTracksP5"),
                       #trajectories   =   cms.InputTag("TrackRefitterP5"),
                       #trajectories = cms.InputTag("CalibrationTracksRefit")
                       trajectories        = cms.InputTag("generalTracks"),
                       siStripClusters     = cms.InputTag("siStripClusters"),
                       siStripDigis        = cms.InputTag("siStripDigis"),
                       trackerEvent        = cms.InputTag("MeasurementTrackerEvent"),
                       lumiScalers = cms.InputTag("scalersRawToDigi"),
                       addLumi = cms.untracked.bool(False),
                       commonMode = cms.InputTag("siStripDigis", "CommonMode"),
                       addCommonMode = cms.untracked.bool(False),
                       # do not cut on the total number of tracks
                       cutOnTracks = cms.untracked.bool(True),
                       # compatibility
                       trackMultiplicity = cms.untracked.uint32(100),
                       # use or not first and last measurement of a trajectory (biases), default is false
                       useFirstMeas = cms.untracked.bool(False),
                       useLastMeas = cms.untracked.bool(False),
                       # use or not all hits when some missing hits in the trajectory (bias), default is false
                       useAllHitsFromTracksWithMissingHits = cms.untracked.bool(False)
                       )

hiteff = cms.Sequence( anEff )
