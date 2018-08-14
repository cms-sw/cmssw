import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
glbMuonSegmentAnalyzer = DQMEDAnalyzer('SegmentTrackAnalyzer',
                                        MuonServiceProxy,
                                        
                                        phiMin = cms.double(-3.2),
                                        ptBin = cms.int32(200),
                                        SegmentsTrackAssociatorParameters = cms.PSet(
                                                         segmentsDt       = cms.untracked.InputTag("dt4DSegments"),
                                                         SelectedSegments = cms.untracked.InputTag("SelectedSegments"),
                                                         segmentsCSC      = cms.untracked.InputTag("cscSegments")
                                                         ),
                                        etaBin = cms.int32(100),
                                        etaMin = cms.double(-3.0),
                                        ptMin = cms.double(0.0),
                                        phiBin = cms.int32(100),
                                        ptMax = cms.double(200.0),
                                        etaMax = cms.double(3.0),
                                        phiMax = cms.double(3.2),

                                        MuTrackCollection = cms.InputTag("globalMuons"),
                                        )

staMuonSegmentAnalyzer = DQMEDAnalyzer('SegmentTrackAnalyzer',
                                         MuonServiceProxy,
                                         
                                         phiMin = cms.double(-3.2),
                                         ptBin = cms.int32(200),
                                         SegmentsTrackAssociatorParameters = cms.PSet(
                                                         segmentsDt       = cms.untracked.InputTag("dt4DSegments"),
                                                         SelectedSegments = cms.untracked.InputTag("SelectedSegments"),
                                                         segmentsCSC      = cms.untracked.InputTag("cscSegments")
                                                         ),
                                         etaBin = cms.int32(100),
                                         etaMin = cms.double(-3.0),
                                         ptMin = cms.double(0.0),
                                         phiBin = cms.int32(100),
                                         ptMax = cms.double(200.0),
                                         etaMax = cms.double(3.0),
                                         phiMax = cms.double(3.2),

                                         MuTrackCollection = cms.InputTag("standAloneMuons"),
                                         )
