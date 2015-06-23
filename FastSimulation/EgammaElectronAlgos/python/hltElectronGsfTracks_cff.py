from FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi

# This should be similar to
# from  FastSimulation.Configuration.StandardSequences_cff import iterativeTrackingBeginning
# from  FastSimulation.Configuration.StandardSequences_cff import famosGsfTrackSequence

hltEgammaCkfTrackCandidatesForGSF = electronGSGsfTrackCandidates.clone()
hltEgammaCkfTrackCandidatesForGSF.src = "hltEgammaElectronPixelSeeds"
hltEgammaGsfTracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
hltEgammaGsfTracks.src = 'hltEgammaCkfTrackCandidatesForGSF'
hltEgammaGsfTracks.TTRHBuilder = 'WithoutRefit'
hltEgammaGsfTracks.TrajectoryInEvent = True

hltEgammaCkfTrackCandidatesForGSFUnseeded = electronGSGsfTrackCandidates.clone()
hltEgammaCkfTrackCandidatesForGSFUnseeded.src = "hltEgammaElectronPixelSeedsUnseeded"
hltEgammaGsfTracksUnseeded = hltEgammaGsfTracks.clone()
hltEgammaGsfTracksUnseeded.src =  'hltEgammaCkfTrackCandidatesForGSFUnseeded'
