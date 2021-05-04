import FWCore.ParameterSet.Config as cms

def producerUseTMTT(process):
    from L1Trigger.TrackerDTC.ProducerED_cfi import TrackerDTCProducer_params
    TrackerDTCProducer_params.UseHybrid = cms.bool( False )
    process.TrackerDTCProducer = cms.EDProducer('trackerDTC::ProducerED', TrackerDTCProducer_params)
    return process

def analyzerUseTMTT(process):
    from L1Trigger.TrackerDTC.Analyzer_cfi import TrackerDTCAnalyzer_params
    from L1Trigger.TrackerDTC.ProducerED_cfi import TrackerDTCProducer_params
    TrackerDTCProducer_params.UseHybrid = cms.bool( False )
    process.TrackerDTCAnalyzer = cms.EDAnalyzer('trackerDTC::Analyzer', TrackerDTCAnalyzer_params, TrackerDTCProducer_params)
    return process