import FWCore.ParameterSet.Config as cms

hiTrackingDqmClientHeavyIons = cms.EDProducer("TrackingDQMClientHeavyIons",
                                              FolderName = cms.string('Tracking/TrackParameters/GeneralProperties')
                                              )
