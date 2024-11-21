# function to manipilate TrackerDTC emulator to match TMTT configuration and support TMTT data formats

import FWCore.ParameterSet.Config as cms

def producerUseTMTT(process):
    from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
    TrackerDTC_params.UseHybrid = False
    process.TrackTriggerSetup.TrackFinding.MinPt = 3.0
    process.TrackTriggerSetup.TrackFinding.MaxEta = 2.4
    process.TrackTriggerSetup.TrackFinding.ChosenRofPhi = 67.24
    process.ProducerDTC = cms.EDProducer('trackerDTC::ProducerDTC', TrackerDTC_params)
    return process

def analyzerUseTMTT(process):
    from L1Trigger.TrackerDTC.Analyzer_cfi import TrackerDTCAnalyzer_params
    from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
    TrackerDTC_params.UseHybrid = False
    process.TrackTriggerSetup.TrackFinding.MinPt = 3.0
    process.TrackTriggerSetup.TrackFinding.MaxEta = 2.4
    process.TrackTriggerSetup.TrackFinding.ChosenRofPhi = 67.24
    process.AnalyzerDTC = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTC_params)
    return process
