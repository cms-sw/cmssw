import FWCore.ParameterSet.Config as cms

shallowGainCalibration = cms.EDProducer("ShallowGainCalibration",
                                      Tracks=cms.InputTag("generalTracks",""),
                                      Prefix=cms.string("GainCalibration"),
                                      Suffix=cms.string(""))
