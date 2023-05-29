import FWCore.ParameterSet.Config as cms

# Use compressiong settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 201

anResol = cms.EDAnalyzer("HitResol",
                         CompressionSettings = cms.untracked.int32(compressionSettings),
                         Debug = cms.bool(False),
                         Layer = cms.int32(0), # = 0 means do all layers
                         #combinatorialTracks = cms.InputTag("ctfWithMaterialTracksP5"),
                         #combinatorialTracks = cms.InputTag("TrackRefitterP5"),
                         #combinatorialTracks = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                         combinatorialTracks = cms.InputTag("generalTracks"),
                         #trajectories = cms.InputTag("ctfWithMaterialTracksP5"),
                         #trajectories   =   cms.InputTag("TrackRefitterP5"),
                         #trajectories = cms.InputTag("CalibrationTracksRefit"),
                         trajectories        = cms.InputTag("generalTracks"),
                         lumiScalers         = cms.InputTag("scalersRawToDigi"),
                         addLumi = cms.untracked.bool(False),
                         # do not cut on the total number of tracks
                         cutOnTracks = cms.untracked.bool(True),
                         # compatibility
                         trackMultiplicity = cms.untracked.uint32(100),
                         MomentumCut = cms.untracked.double(3.),
                         UsePairsOnly = cms.untracked.uint32(1))

hitresol = cms.Sequence( anResol )
