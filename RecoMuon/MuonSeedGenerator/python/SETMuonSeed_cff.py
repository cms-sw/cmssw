import FWCore.ParameterSet.Config as cms

# Magnetic Field
# Geometries
from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
#import TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi
#KFTrajectoryFitterForSTA = TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi.KFTrajectoryFitter.clone()
# Stand Alone Muons Producer
from RecoMuon.MuonSeedGenerator.SETMuonSeed_cfi import *



