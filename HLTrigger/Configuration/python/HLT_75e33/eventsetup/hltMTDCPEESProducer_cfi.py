import FWCore.ParameterSet.Config as cms

hltMTDCPEESProducer = cms.ESProducer('MTDCPEESProducer',
                                     appendToDataLabel = cms.string('')
                                     )
