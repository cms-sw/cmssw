import FWCore.ParameterSet.Config as cms

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackGLBMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackInnerTrackMuons_cff import *
from DQMOffline.Muon.dtSegmTask_cfi import *

#dedicated analyzers for offline dqm 
from DQMOffline.Muon.muonAnalyzer_cff import *
from DQMOffline.Muon.CSCMonitor_cfi import *
from DQMOffline.Muon.gemOfflineMonitor_cfi import *
from DQMOffline.Muon.gemEfficiencyAnalyzer_cfi import *
from DQMOffline.Muon.muonIdDQM_cff import *
from DQMOffline.Muon.muonIsolationDQM_cff import *

#dedicated clients for offline dqm 
from DQMOffline.Muon.muonQualityTests_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoMuons = DQMEDAnalyzer('DQMEventInfo',
                              subSystemFolder = cms.untracked.string('Muons')
                              )

muonTrackAnalyzers = cms.Sequence(MonitorTrackSTAMuons*MonitorTrackGLBMuons*MonitorTrackINNMuons)

muonMonitors = cms.Sequence(muonTrackAnalyzers*
                            dtSegmentsMonitor*
                            cscMonitor*
                            muonAnalyzer*
                            muonIdDQM*
                            dqmInfoMuons*
                            muIsoDQM_seq)

muonMonitors_miniAOD = cms.Sequence( muonAnalyzer_miniAOD*
                                     muIsoDQM_seq_miniAOD)


muonMonitorsAndQualityTests = cms.Sequence(muonMonitors*muonQualityTests)

_run3_muonMonitors = muonMonitors.copy()
_run3_muonMonitors += gemOfflineMonitor
_run3_muonMonitors += gemEfficiencyAnalyzerSeq

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(muonMonitors, _run3_muonMonitors)
