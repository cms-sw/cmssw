# EDAnalyzer for Track Trigger emulation steps

import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.Analyzer_cfi import TrackerTFPAnalyzer_params
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params

AnalyzerGP  = cms.EDAnalyzer( 'trackerTFP::AnalyzerGP' , TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
AnalyzerHT  = cms.EDAnalyzer( 'trackerTFP::AnalyzerHT' , TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
AnalyzerCTB = cms.EDAnalyzer( 'trackerTFP::AnalyzerCTB', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
AnalyzerKF  = cms.EDAnalyzer( 'trackerTFP::AnalyzerKF' , TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
AnalyzerDR  = cms.EDAnalyzer( 'trackerTFP::AnalyzerDR' , TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
