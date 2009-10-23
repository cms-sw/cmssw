import FWCore.ParameterSet.Config as cms

anEff = cms.EDAnalyzer("HitEff",
                       Debug = cms.bool(False),
                       Layer = cms.int32(0), # =0 means do all layers
                       #combinatorialTracks = cms.InputTag("ctfWithMaterialTracksP5"),
                       #combinatorialTracks = cms.InputTag("TrackRefitterP5"),
                       combinatorialTracks = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                       #trajectories = cms.InputTag("ctfWithMaterialTracksP5"),
                       #trajectories   =   cms.InputTag("TrackRefitterP5"),
                       trajectories = cms.InputTag("CalibrationTracksRefit")
                       )

hiteff = cms.Sequence( anEff )
