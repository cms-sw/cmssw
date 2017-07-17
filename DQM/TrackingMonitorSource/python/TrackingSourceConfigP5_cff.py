import FWCore.ParameterSet.Config as cms

# TrackingMonitor ####
# Clone for Cosmic Track Finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName      = 'CosmicTk'
TrackMon_cosmicTk.FolderName    = 'Tracking/TrackParameters'
TrackMon_cosmicTk.doSeedParameterHistos = True
TrackMon_cosmicTk.TkSizeBin = cms.int32(4)
TrackMon_cosmicTk.TkSizeMax = cms.double( 3.5)
TrackMon_cosmicTk.TkSizeMin = cms.double(-0.5)

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName           = 'CKFTk'
TrackMon_ckf.FolderName         = 'Tracking/TrackParameters'
TrackMon_ckf.doSeedParameterHistos = True
TrackMon_ckf.TkSizeBin = cms.int32(4)
TrackMon_ckf.TkSizeMax = cms.double( 3.5)
TrackMon_ckf.TkSizeMin = cms.double(-0.5)

# Clone for Road Search  Tracks
#import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
#TrackMon_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
#TrackMon_rs.TrackProducer       = 'rsWithMaterialTracksP5'
#TrackMon_rs.AlgoName            = 'RSTk'
#TrackMon_rs.FolderName          = 'Tracking/TrackParameters'
#TrackMon_rs.doSeedParameterHistos = True

# Clone for General Track (for Collision data)
import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackMon_gentk = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackMon_gentk.FolderName          = 'Tracking/TrackParameters'
TrackMon_gentk.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
# decrease number of histograms
#TrackMon_gentk.doTrackerSpecific = False

# Clone for Heavy Ion Tracks (for HI Collisions)
import DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi
TrackMon_hi = DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi.TrackerHeavyIonTrackMon.clone()
TrackMon_hi.FolderName          = 'Tracking/TrackParameters'
TrackMon_hi.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'

# Tracking Efficiency ####
# Clone for Cosmic Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_cosmicTk = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_cosmicTk.TKTrackCollection             = 'cosmictrackfinderP5'
TrackEffMon_cosmicTk.AlgoName                      = 'CosmicTk'
TrackEffMon_cosmicTk.FolderName                    = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_ckf = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_ckf.TKTrackCollection                  = 'ctfWithMaterialTracksP5'
TrackEffMon_ckf.AlgoName                           = 'CKFTk'
TrackEffMon_ckf.FolderName                         = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for RS Tracks
#import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
#TrackEffMon_rs = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
#TrackEffMon_rs.TKTrackCollection                   = 'rsWithMaterialTracksP5'
#TrackEffMon_rs.AlgoName                            = 'RSTk'
#TrackEffMon_rs.FolderName                          = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for Beam Halo  Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_bhmuon = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_bhmuon.TKTrackCollection               = 'ctfWithMaterialTracksBeamHaloMuon'
TrackEffMon_bhmuon.AlgoName                        = 'BHMuonTk'
TrackEffMon_bhmuon.FolderName                      = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for Heavy Ion Tracks (for HI Collisions)
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_hi = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_hi.TKTrackCollection                   = 'hiGeneralTracks'
TrackEffMon_hi.AlgoName                            = 'HeavyIonTk'
TrackEffMon_hi.FolderName                          = 'Tracking/TrackParameters/TrackEfficiency'
