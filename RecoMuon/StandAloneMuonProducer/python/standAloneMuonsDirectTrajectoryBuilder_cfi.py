import FWCore.ParameterSet.Config as cms

# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *

standAloneMuons = cms.EDProducer("StandAloneMuonProducer",
                                 MuonTrackLoaderForSTA,
                                 MuonServiceProxy,
                                 InputObjects = cms.InputTag("MuonSeed"),
                                 MuonTrajectoryBuilder = cms.string("DirectMuonTrajectoryBuilder"),
                                 STATrajBuilderParameters = cms.PSet(SeedTransformerParameters = cms.PSet(Fitter = cms.string('KFFitterSmootherSTA'),
                                                                                                          RescaleError = cms.double(100.0),
                                                                                                          MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                                                                                          Propagator = cms.string('SteppingHelixPropagatorAny'),
                                                                                                          NMinRecHits = cms.uint32(2),
                                                                                                          UseSubRecHits = cms.bool(True)
                                                                                                          )
                                                                     )
                                 )



