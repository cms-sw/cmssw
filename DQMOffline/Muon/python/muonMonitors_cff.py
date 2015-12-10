import FWCore.ParameterSet.Config as cms

#
# This object is used to make changes for different running scenarios
#
from Configuration.StandardSequences.Eras import eras

#Analyzer taken from online dqm
from DQM.TrackingMonitor.MonitorTrackSTAMuons_cfi import *
from DQM.TrackingMonitor.MonitorTrackGLBMuons_cfi import *
from DQMOffline.Muon.dtSegmTask_cfi import *

#dedicated analyzers for offline dqm 
from DQMOffline.Muon.muonAnalyzer_cff import *
from DQMOffline.Muon.CSCMonitor_cfi import *
from DQMOffline.Muon.muonIdDQM_cff import *
from DQMOffline.Muon.muonIsolationDQM_cff import *

#dedicated clients for offline dqm 
from DQMOffline.Muon.muonQualityTests_cff import *

dqmInfoMuons = cms.EDAnalyzer("DQMEventInfo",
                              subSystemFolder = cms.untracked.string('Muons')
                              )

muonTrackAnalyzers = cms.Sequence(MonitorTrackSTAMuons*MonitorTrackGLBMuons)

muonMonitors = cms.Sequence(muonTrackAnalyzers*
                            dtSegmentsMonitor*
                            cscMonitor*
                            muonAnalyzer*
                            muonIdDQM*
                            dqmInfoMuons*
                            muIsoDQM_seq)
# Modify for if the phase 1 pixel detector is active
if eras.phase1Pixel.isChosen() :
    muonMonitors.remove(muonAnalyzer)

muonMonitorsAndQualityTests = cms.Sequence(muonMonitors*muonQualityTests)


