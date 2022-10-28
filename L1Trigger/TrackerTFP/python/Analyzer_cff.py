import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.Analyzer_cfi import TrackerTFPAnalyzer_params
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params

TrackerTFPAnalyzerGP = cms.EDAnalyzer( 'trackerTFP::AnalyzerGP', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerHT = cms.EDAnalyzer( 'trackerTFP::AnalyzerHT', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerMHT = cms.EDAnalyzer( 'trackerTFP::AnalyzerMHT', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerZHT = cms.EDAnalyzer( 'trackerTFP::AnalyzerZHT', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerKFin = cms.EDAnalyzer( 'trackerTFP::AnalyzerKFin', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerKF = cms.EDAnalyzer( 'trackerTFP::AnalyzerKF', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )
TrackerTFPAnalyzerTT = cms.EDAnalyzer( 'trackerTFP::AnalyzerTT', TrackerTFPAnalyzer_params, TrackerTFPProducer_params )