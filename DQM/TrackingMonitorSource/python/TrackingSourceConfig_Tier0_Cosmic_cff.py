import FWCore.ParameterSet.Config as cms

# TrackingMonitor ####
# Clone for Cosmic Track Finder
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_cosmicTk = TrackerCosmicTrackMon.clone(
    TrackProducer = 'cosmictrackfinderP5',
    AlgoName = 'CosmicTk',
    FolderName = 'Tracking/TrackParameters',
    doSeedParameterHistos = True
)

# Clone for CKF Tracks
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_ckf = TrackerCosmicTrackMon.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    AlgoName = 'CKFTk',
    FolderName = 'Tracking/TrackParameters',
    doSeedParameterHistos = True
)

# Clone for Road Search  Tracks
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMon_rs = TrackerCosmicTrackMon.clone(
    TrackProducer = 'rsWithMaterialTracksP5',
    AlgoName = 'RSTk',
    FolderName = 'Tracking/TrackParameters',
    doSeedParameterHistos = True
)

# Clone for Beam Halo Muon Tracks
# from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
# TrackMon_bhmuon = TrackerCosmicTrackMon.clone(
#     TrackProducer = 'ctfWithMaterialTracksBeamHaloMuon',
#     AlgoName = 'BHMuonTk',
#     FolderName = 'Tracking/TrackParameters',
#     doSeedParameterHistos = True
# )

# Tracking Efficiency
# Clone for Cosmic Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_cosmicTk = TrackEffMon.clone( 
    TKTrackCollection = 'cosmictrackfinderP5',
    AlgoName = 'CosmicTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Clone for CKF Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_ckf = TrackEffMon.clone( 
    TKTrackCollection = 'ctfWithMaterialTracksP5',
    AlgoName = 'CKFTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Clone for RS Tracks
# from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
# TrackEffMon_rs = TrackEffMon.clone( 
#     TKTrackCollection = 'rsWithMaterialTracksP5',
#     AlgoName = 'RSTk',
#     FolderName = 'Tracking/TrackParameters/TrackEfficiency'
# )

# Clone for Beam Halo  Tracks
from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import *
TrackEffMon_bhmuon = TrackEffMon.clone( 
    TKTrackCollection = 'ctfWithMaterialTracksBeamHaloMuon',
    AlgoName = 'BHMuonTk',
    FolderName = 'Tracking/TrackParameters/TrackEfficiency'
)

# Split Tracking
from  DQM.TrackingMonitor.TrackSplittingMonitor_cfi import *
TrackSplitMonitor.FolderName = 'Tracking/TrackParameters/SplitTracks'

# LogMessageMonitor ####
### load which are the module to monitor
from DQM.TrackingMonitorSource.EDModules2monitor_cfi import *

### load the different flavour of settings of the LogMessageMonitor module
from DQM.TrackingMonitorSource.LogMessageMonitor_cff import *

for module in selectedModules4cosmics :
    label = str(module)+'LogMessageMon'
    locals()[label] = LogMessageMonCommon.clone(
        pluginsMonName = pluginsMonName[module],
        modules = modulesLabel[module],
        categories = categories[module]
    )
    locals()[label].setLabel(label)

# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoTracking = DQMEDAnalyzer('DQMEventInfo',
     subSystemFolder = cms.untracked.string('Tracking')
)

# Sequences 
TrackingDQMTier0_cosmicTk = cms.Sequence(TrackMon_cosmicTk*TrackEffMon_cosmicTk)

TrackingDQMTier0_ckf = cms.Sequence(TrackMon_ckf*TrackEffMon_ckf)

#TrackingDQMTier0_rs = cms.Sequence(TrackMon_rs*TrackEffMon_rs)

#TrackingDQMTier0 = cms.Sequence(TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoTracking)
TrackingDQMTier0 = cms.Sequence(TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor)

# MessageLog
for module in selectedModules4cosmics :
    label = str(module)+'LogMessageMon'
    TrackingDQMTier0_cosmicTk += cms.Sequence(locals()[label])
    TrackingDQMTier0_ckf      += cms.Sequence(locals()[label])
    TrackingDQMTier0          += cms.Sequence(locals()[label])

TrackingDQMTier0 += cms.Sequence(dqmInfoTracking)
