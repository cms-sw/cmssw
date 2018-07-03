import FWCore.ParameterSet.Config as cms

from DQMOffline.Hcal.HLTHcalRecHitParam_cfi import *

hcalMonitoringSequence = cms.Sequence(
    hltHCALRecHitsAnalyzer
)
