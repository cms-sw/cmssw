import FWCore.ParameterSet.Config as cms

# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *

standAloneMuons = cms.EDProducer("StandAloneMuonProducer",
                                 MuonTrackLoaderForSTA,
                                 MuonServiceProxy,
#                                 InputObjects = cms.InputTag("mergedStandAloneMuonSeeds"),
                                 InputObjects = cms.InputTag("ancientMuonSeed"),
                                 MuonTrajectoryBuilder = cms.string("StandAloneMuonTrajectoryBuilder"),
                                 STATrajBuilderParameters = cms.PSet(NavigationType = cms.string('Standard'),
                                                                     SeedPosition = cms.string('in'),
                                                                     SeedPropagator = cms.string('SteppingHelixPropagatorAny'),

                                                                     DoSeedRefit = cms.bool(False),
                                                                     SeedTransformerParameters = cms.PSet(Fitter = cms.string('KFFitterSmootherSTA'),
                                                                                                          RescaleError = cms.double(100.0),
                                                                                                          MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                                                                                          Propagator = cms.string('SteppingHelixPropagatorAny'),
                                                                                                          NMinRecHits = cms.uint32(2),
                                                                                                          UseSubRecHits = cms.bool(False)
                                                                                                          ),
    
                                                                     FilterParameters = cms.PSet(FitDirection = cms.string('insideOut'),
                                                                                                 EnableDTMeasurement = cms.bool(True),
                                                                                                 DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
                                                                                                 EnableCSCMeasurement = cms.bool(True),
                                                                                                 CSCRecSegmentLabel = cms.InputTag("cscSegments"),
                                                                                                 EnableRPCMeasurement = cms.bool(True),
                                                                                                 RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
                                                                                                 NumberOfSigma = cms.double(3.0),
                                                                                                 MaxChi2 = cms.double(1000.0),
                                                                                                 Propagator = cms.string('SteppingHelixPropagatorAny'),
                                                                                                 MuonTrajectoryUpdatorParameters = cms.PSet(MaxChi2 = cms.double(1000.0),
                                                                                                                                            RescaleError = cms.bool(False),
                                                                                                                                            RescaleErrorFactor = cms.double(100.0),
                                                                                                                                            Granularity = cms.int32(0)
                                                                                                                                            )
                                                                                                 ),
      
                                                                     DoBackwardFilter = cms.bool(True),
                                                                     BWFilterParameters = cms.PSet(FitDirection = cms.string('outsideIn'),
                                                                                                   BWSeedType = cms.string('fromGenerator'),
                                                                                                   EnableDTMeasurement = cms.bool(True),
                                                                                                   DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
                                                                                                   EnableCSCMeasurement = cms.bool(True),
                                                                                                   CSCRecSegmentLabel = cms.InputTag("cscSegments"),
                                                                                                   EnableRPCMeasurement = cms.bool(True),
                                                                                                   RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
                                                                                                   NumberOfSigma = cms.double(3.0),
                                                                                                   MaxChi2 = cms.double(100.0),
                                                                                                   Propagator = cms.string('SteppingHelixPropagatorAny'),
                                                                                                   MuonTrajectoryUpdatorParameters = cms.PSet(MaxChi2 = cms.double(100.0),
                                                                                                                                              RescaleError = cms.bool(False),
                                                                                                                                              RescaleErrorFactor = cms.double(100.0),
                                                                                                                                              Granularity = cms.int32(2)
                                                                                                                                              )
                                                                                                   ),
                                                                     DoRefit = cms.bool(False),
                                                                     RefitterParameters = cms.PSet(FitterName = cms.string('KFFitterSmootherSTA'),
                                                                                                   NumberOfIterations = cms.uint32(3),
                                                                                                   ForceAllIterations = cms.bool(False),
                                                                                                   MaxFractionOfLostHits = cms.double(0.05),
                                                                                                   RescaleError = cms.double(100.)
                                                                                                   )
                                                                     
                                                                     )
                                 )



