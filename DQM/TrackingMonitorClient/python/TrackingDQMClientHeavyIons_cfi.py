import FWCore.ParameterSet.Config as cms

hiTrackingDqmClientHeavyIons = cms.EDAnalyzer("TrackingDQMClientHeavyIons",
                                              FolderName = cms.string('Tracking/TrackParameters/GeneralProperties')
                                              )
