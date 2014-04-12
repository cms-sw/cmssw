import FWCore.ParameterSet.Config as cms

cosmictrackSelector = cms.EDProducer("CosmicTrackSelector",
                                     src = cms.InputTag("ctfWithMaterialTracksCosmics"),
                                     keepAllTracks = cms.bool(False), ## if set to true tracks failing this filter are kept in the output
                                     beamspot = cms.InputTag("offlineBeamSpot"),
                                     #untracked bool copyTrajectories = true // when doing retracking before
                                     copyTrajectories = cms.untracked.bool(True),
                                     copyExtras = cms.untracked.bool(True), ## set to false on AOD
                                     qualityBit = cms.string(''), # set to '' or comment out if you don't want to set the bit                                     
                                     # parameters for adapted optimal cuts on chi2
                                     chi2n_par = cms.double(10.0),
                                     # Impact parameter absolute cuts.
                                     max_d0 = cms.double(110.),
                                     max_z0 = cms.double(300.),
                                     # track parameter cuts 
                                     max_eta = cms.double(2.0),
                                     min_pt = cms.double(1.0),
                                     # Cut on numbers of valid hits
                                     min_nHit = cms.uint32(5),
                                     # Cut on number of Pixel Hit 
                                     min_nPixelHit = cms.uint32(0),
                                     # Cuts on numbers of layers with hits/3D hits/lost hits. 
                                     minNumberLayers = cms.uint32(0),
                                     minNumber3DLayers = cms.uint32(0),
                                     maxNumberLostLayers = cms.uint32(999)
                                     )


