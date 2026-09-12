import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.AnalyzerMC_cfi import TrackTriggerAnalyzerMC_params
from SimTracker.TrackTriggerAssociation.StubAssociator_cfi import StubAssociator_params

AnalyzerMC = cms.EDAnalyzer( 'tt::AnalyzerMC', TrackTriggerAnalyzerMC_params, StubAssociator_params )
