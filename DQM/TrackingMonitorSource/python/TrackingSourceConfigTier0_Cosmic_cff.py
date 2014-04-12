import FWCore.ParameterSet.Config as cms

# TrackingMonitor ####
# Clone for Cosmic Track Finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_cosmicTk.TrackProducer                    = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName                         = 'CosmicTk'
TrackMon_cosmicTk.FolderName                       = 'Tracking/TrackParameters'
TrackMon_cosmicTk.doSeedParameterHistos            = True

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_ckf.TrackProducer                         = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName                              = 'CKFTk'
TrackMon_ckf.FolderName                            = 'Tracking/TrackParameters'
TrackMon_ckf.doSeedParameterHistos                 = True

# Clone for Road Search  Tracks
#import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
#TrackMon_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
#TrackMon_rs.TrackProducer                          = 'rsWithMaterialTracksP5'
#TrackMon_rs.AlgoName                               = 'RSTk'
#TrackMon_rs.FolderName                             = 'Tracking/TrackParameters'
#TrackMon_rs.doSeedParameterHistos                  = True

# Clone for Beam Halo Muon Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_bhmuon = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_bhmuon.TrackProducer                      = 'ctfWithMaterialTracksBeamHaloMuon'
TrackMon_bhmuon.AlgoName                           = 'BHMuonTk'
TrackMon_bhmuon.FolderName                         = 'Tracking/TrackParameters'
TrackMon_bhmuon.doSeedParameterHistos              = True

# Tracking Efficiency
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

# Split Tracking
from  DQM.TrackingMonitor.TrackSplittingMonitor_cfi import *
TrackSplitMonitor.FolderName = 'Tracking/TrackParameters/SplitTracks'


# DQM Services
dqmInfoTracking = cms.EDAnalyzer("DQMEventInfo",
     subSystemFolder = cms.untracked.string('Tracking')
)

# Sequences 
TrackingDQMTier0_cosmicTk = cms.Sequence(TrackMon_cosmicTk*TrackEffMon_cosmicTk)

TrackingDQMTier0_ckf = cms.Sequence(TrackMon_ckf*TrackEffMon_ckf)

#TrackingDQMTier0_rs = cms.Sequence(TrackMon_rs*TrackEffMon_rs)

TrackingDQMTier0 = cms.Sequence(TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoTracking)
