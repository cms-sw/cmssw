import FWCore.ParameterSet.Config as cms
gmtTkMuons = cms.EDProducer('Phase2L1TGMTTkMuonProducer',
                     srcTracks = cms.InputTag("l1tTTTracksFromTrackletEmulation:Level1TTTracks"),
                     srcStubs  = cms.InputTag('gmtStubs:tps'),
                     minTrackStubs = cms.int32(4),     
                     muonBXMin = cms.int32(0),
                     muonBXMax = cms.int32(0),
                            verbose   = cms.int32(0),     
                     trackConverter  = cms.PSet(
                         verbose = cms.int32(0)
                     ),
                     trackMatching  = cms.PSet(
                         verbose=cms.int32(0)
                     ),
                     isolation  = cms.PSet(
                       AbsIsoThresholdL = cms.int32(160),
                       AbsIsoThresholdM = cms.int32(120),
                       AbsIsoThresholdT = cms.int32(80),
                       RelIsoThresholdL = cms.double(0.1),
                       RelIsoThresholdM = cms.double(0.05),
                       RelIsoThresholdT = cms.double(0.01),
                       verbose       = cms.int32(0),
                       IsodumpForHLS = cms.int32(0),
                     ),
                    tauto3mu = cms.PSet()

)





