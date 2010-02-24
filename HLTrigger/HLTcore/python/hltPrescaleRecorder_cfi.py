import FWCore.ParameterSet.Config as cms

hltPrescaleRecorder = cms.EDProducer("HLTPrescaleRecorder",
    src        = cms.int32(0),
    run        = cms.bool(True),
    lumi       = cms.bool(True),
    event      = cms.bool(True),
    hltInputTag= cms.InputTag("","","")                            
)
