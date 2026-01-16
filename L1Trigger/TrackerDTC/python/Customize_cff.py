# function to manipilate TrackerDTC emulator to match TMTT configuration and support TMTT data formats

import FWCore.ParameterSet.Config as cms

def setupTMTT(process):
    from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
    # use Hybrid or TMTT as TT algorithm
    process.TrackTriggerSetup.UseHybrid = False
    # min track pt in GeV, also defines region overlap shape
    process.TrackTriggerSetup.TrackFinding.MinPt = 3.0
    # cut on stub eta
    process.TrackTriggerSetup.TrackFinding.MaxEta = 2.4
    # critical radius defining region overlap shape in cm
    process.TrackTriggerSetup.TrackFinding.ChosenRofPhi = 67.24


def producerUseTMTT(process):
    from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
    setupTMTT(process)
    process.ProducerDTC = cms.EDProducer('trackerDTC::ProducerDTC', TrackerDTC_params)
    return process

def analyzerUseTMTT(process):
    from L1Trigger.TrackerDTC.Analyzer_cfi import TrackerDTCAnalyzer_params
    from L1Trigger.TrackerDTC.DTC_cfi import TrackerDTC_params
    setupTMTT(process)
    process.AnalyzerDTC = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTC_params)
    return process
