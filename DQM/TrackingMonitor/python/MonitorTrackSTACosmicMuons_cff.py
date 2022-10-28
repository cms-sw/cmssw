import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackSTACosmicMuonsBarrel = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/cosmicMuonsBarrelOnly',
    TrackProducer = 'cosmicMuonsBarrelOnly'
)
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackSTACosmicMuonsEndCaps = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/cosmicMuonsEndCapsOnly',
    TrackProducer = 'cosmicMuonsEndCapsOnly'
)
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackSTACosmicMuons1LegBarrel = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/cosmicMuons1LegBarrelOnly',
    TrackProducer = 'cosmicMuons1LegBarrelOnly'
)
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackLHCStandAloneMuonsBarrelOnly = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/standAloneMuonsBarrelOnly',
    TrackProducer = 'lhcStandAloneMuonsBarrelOnly'
)
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackLHCStandAloneMuonsEndCapsOnly = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/standAloneMuonsEndCapsOnly',
    TrackProducer = 'lhcStandAloneMuonsEndCapsOnly'
)
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
MonitorTrackSTACosmicMuonsNoDriftBarrel = MonitorTrackSTAMuons.clone(
    FolderName = 'Muons/cosmicMuonsNoDriftBarrelOnly',
    TrackProducer = 'cosmicMuonsNoDriftBarrelOnly'
)
standAloneCosmicMuonsMonitors = cms.Sequence(MonitorTrackSTACosmicMuons*MonitorTrackSTACosmicMuonsBarrel*MonitorTrackSTACosmicMuons1LegBarrel*MonitorTrackSTACosmicMuonsEndCaps*MonitorTrackLHCStandAloneMuonsBarrelOnly*MonitorTrackLHCStandAloneMuonsEndCapsOnly*MonitorTrackSTACosmicMuonsNoDriftBarrel)
