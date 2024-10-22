import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DiMuonVertexMonitor = DQMEDAnalyzer('DiMuonVertexMonitor',
                                    muonTracks = cms.InputTag('ALCARECOTkAlDiMuon'),
                                    decayMotherName = cms.string('Z'),
                                    vertices = cms.InputTag('offlinePrimaryVertices'),
                                    FolderName = cms.string('DiMuonVertexMonitor'),
                                    maxSVdist = cms.double(50),
                                    CosPhi3DConfig = cms.PSet(
                                        name = cms.string('CosPhi3D'),
                                        title = cms.string('cos(#phi_{3D})'),
                                        yUnits = cms.string(''),
                                        NxBins = cms.int32(24),
                                        NyBins = cms.int32(50),
                                        ymin = cms.double(-1),
                                        ymax = cms.double(1),
                                        maxDeltaEta = cms.double(3.7)
                                    ),
                                    SVDistConfig = cms.PSet(
                                        name = cms.string('SVDist'),
                                        title = cms.string('PV-SV distance'),
                                        yUnits = cms.string('[#mum]'),
                                        NxBins = cms.int32(24),
                                        NyBins = cms.int32(100),
                                        ymin = cms.double(0),
                                        ymax = cms.double(300),
                                        maxDeltaEta = cms.double(3.7)
                                    ),
                                    SVDistSigConfig = cms.PSet(
                                        name = cms.string('SVDistSig'),
                                        title = cms.string('PV-SV distance significance'),
                                        yUnits = cms.string('[#mum]'),
                                        NxBins = cms.int32(24),
                                        NyBins = cms.int32(100),
                                        ymin = cms.double(0),
                                        ymax = cms.double(5),
                                        maxDeltaEta = cms.double(3.7)
                                    ),
                                    SVDist3DConfig = cms.PSet(
                                        name = cms.string('SVDist3D'),
                                        title = cms.string('PV-SV 3D distance'),
                                        yUnits = cms.string('[#mum]'),
                                        NxBins = cms.int32(24),
                                        NyBins = cms.int32(100),
                                        ymin = cms.double(0),
                                        ymax = cms.double(300),
                                        maxDeltaEta = cms.double(3.7)
                                    ),
                                    SVDist3DSigConfig = cms.PSet(
                                        name = cms.string('SVDist3DSig'),
                                        title = cms.string('PV-SV 3D distance significance'),
                                        yUnits = cms.string('[#mum]'),
                                        NxBins = cms.int32(24),
                                        NyBins = cms.int32(100),
                                        ymin = cms.double(0),
                                        ymax = cms.double(5),
                                        maxDeltaEta = cms.double(3.7)
                                    ))
