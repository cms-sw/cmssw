import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Scouting.ScoutingEGammaCollectionMonitoring_cfi import *
from HLTriggerOffline.Scouting.ScoutingElectronTagProbeAnalyzer_cfi import *

ScoutingElectronMonitoring = cms.Sequence(scoutingMonitoringEGMOnline + scoutingMonitoringTagProbeOnline)
