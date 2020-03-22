def useTMTT(process):
    from L1Trigger.TrackerDTC.Producer_Defaults_cfi import TrackerDTCProducer_params
    from L1Trigger.TrackerDTC.Format_TMTT_cfi import TrackerDTCFormat_params
    TrackerDTCProducer_params.ParamsED.DataFormat = "TMTT"
    process.TrackerDTCProducer = cms.EDProducer('TrackerDTCProducer', TrackerDTCProducer_params, TrackerDTCFormat_params )
    return process