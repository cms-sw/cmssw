import FWCore.ParameterSet.Config as cms

trackdnn_source = cms.ESSource("EmptyESSource", 
    recordName = cms.string("TfGraphRecord"), 
    firstValid = cms.vuint32(1), 
    iovIsRunNotTime = cms.bool(True)
)
