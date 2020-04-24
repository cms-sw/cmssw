from FastSimulation.Tracking.electronCkfTrackCandidates_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi

hltEgammaCkfTrackCandidatesForGSF = electronCkfTrackCandidates.clone()
hltEgammaCkfTrackCandidatesForGSF.src = "hltEgammaElectronPixelSeeds"
hltEgammaGsfTracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
hltEgammaGsfTracks.src = 'hltEgammaCkfTrackCandidatesForGSF'
hltEgammaGsfTracks.TTRHBuilder = 'WithoutRefit'
hltEgammaGsfTracks.TrajectoryInEvent = True

hltEgammaCkfTrackCandidatesForGSFUnseeded = electronCkfTrackCandidates.clone()
hltEgammaCkfTrackCandidatesForGSFUnseeded.src = "hltEgammaElectronPixelSeedsUnseeded"
hltEgammaGsfTracksUnseeded = hltEgammaGsfTracks.clone()
hltEgammaGsfTracksUnseeded.src =  'hltEgammaCkfTrackCandidatesForGSFUnseeded'
