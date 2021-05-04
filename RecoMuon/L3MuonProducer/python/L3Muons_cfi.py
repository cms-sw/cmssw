import FWCore.ParameterSet.Config as cms
import RecoMuon.L3MuonProducer.L3MuonProducer_cfi as _mod
#this is a dump of the latest configuration of that module
#this is not the actual configuration of HLT
#changing this file will not change the behavior of HLT
#see the actual configuration in confDB

L3Muons = _mod.L3MuonProducer.clone(
    ServiceParameters = dict(
        Propagators = ['SmartPropagatorAny',
            'SteppingHelixPropagatorAny',
            'SmartPropagator',
            'SteppingHelixPropagatorOpposite'],
    ),
    TrackLoaderParameters = dict(
        PutTkTrackIntoEvent = True,
        Smoother = 'KFSmootherForMuonTrackLoader',
        MuonUpdatorAtVertexParameters = dict(
            Propagator = 'SteppingHelixPropagatorOpposite',
        ),
        DoSmoothing = True,
        beamSpot = "hltOfflineBeamSpot"
    ),
    L3TrajBuilderParameters = dict(
        TrackerRecHitBuilder = 'WithTrackAngle',
        MuonTrackingRegionBuilder = dict(
            Rescale_Dz = 3.0,
            Eta_min = 0.05,
            DeltaZ_Region = cms.double(15.9),
            DeltaR = 0.2,
            UseFixedRegion = cms.bool(False),
            Phi_min = 0.05,
            beamSpot = "hltOfflineBeamSpot"
        ),
        GlobalMuonTrackMatcher = dict(
            Propagator = 'SmartPropagator'
        ),
        tkTrajLabel = "hltL3TkTracksFromL2",
	tkTrajBeamSpot = "hltOfflineBeamSpot", # add a filter for L3 trajectory
	tkTrajMaxChi2 = 999, # add a filter for L3 trajectory
	tkTrajMaxDXYBeamSpot = 999, # add a filter for L3 trajectory
	tkTrajVertex = "pixelVertices", # add a filter for L3 trajectory
	tkTrajUseVertex = False, # add a filter for L3 trajectory
        MuonRecHitBuilder = 'MuonRecHitBuilder',
        TrackTransformer = dict(
            Fitter = 'L3MuKFFitter',
            TrackerRecHitBuilder = 'WithTrackAngle',
            Smoother = 'KFSmootherForMuonTrackLoader',
            MuonRecHitBuilder = 'MuonRecHitBuilder',
            Propagator = 'SmartPropagatorAny'
        ),
        GlbRefitterParameters = dict(
            Fitter = 'L3MuKFFitter',
            TrackerRecHitBuilder = 'WithTrackAngle',
            MuonRecHitBuilder = 'MuonRecHitBuilder',
            Propagator = 'SmartPropagatorAny',
        )
    )
)
