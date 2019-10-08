#Analyzer taken from online dqm

from DQM.TrackingMonitor.TrackingMonitor_cfi import *
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuons_cff import *
from DQM.TrackingMonitor.MonitorTrackGLBCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackTKCosmicMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackSTACosmicMuonsHLT_cff import *
from DQM.MuonMonitor.muonCosmicAnalyzer_cff import * 

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *




from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoMuons = DQMEDAnalyzer('DQMEventInfo',
                             subSystemFolder = cms.untracked.string('Muons')
)


muonTrackCosmicAnalyzers= cms.Sequence(MonitorTrackSTACosmicMuons*
                                       MonitorTrackTKCosmicMuons *
                                       MonitorTrackGLBCosmicMuons)


muonCosmicMonitors = cms.Sequence(muonTrackCosmicAnalyzers* 
                                  muonCosmicAnalyzer)




