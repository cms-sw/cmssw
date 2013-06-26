import FWCore.ParameterSet.Config as cms

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# TrackProducer
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
findSiElectrons = cms.EDProducer("SiStripElectronProducer",
    siStereoHitCollection = cms.string('stereoRecHit'),
    maxHitsOnDetId = cms.int32(4),
    minHits = cms.int32(5),
    trackCandidatesLabel = cms.string(''),
    superClusterProducer = cms.string('correctedHybridSuperClusters'),
    phiBandWidth = cms.double(0.01), ## radians

    siStripElectronsLabel = cms.string('findSiElectronsInSiStrips'),
    siRphiHitCollection = cms.string('rphiRecHit'),
    siHitProducer = cms.string('siStripMatchedRecHits'),
    maxReducedChi2 = cms.double(10000.0), ## might not work yet

    originUncertainty = cms.double(15.0), ## cm

    maxNormResid = cms.double(10.0),
    siMatchedHitCollection = cms.string('matchedRecHit'),
    superClusterCollection = cms.string('')
)

# the above produces this warning in 1_3_0_pre1
# WARNING: do not embed replace statements to modify a parameter from a module which hasn't been cloned: 
#  Parameter src in ctfWithMaterialTracks
#  Replace happens in RecoEgamma/EgammaElectronProducers/data/test13_code.cfi
#  This will be an error in future releases.  Please fix.
#module siElectronCtfWithMaterialTracks = TrackProducer 
#{
#  string Fitter = "KFFittingSmoother"   
#  string Propagator = "PropagatorWithMaterial" 
#  string src ="findSiElectrons"
#  string producer = ""
#  string TTRHBuilder       =   "WithTrackAngle"
#  bool TrajectoryInEvent = false
#} 
associateSiElectronsWithTracks = cms.EDProducer("SiStripElectronAssociator",
    siStripElectronCollection = cms.string('findSiElectronsInSiStrips'),
    trackCollection = cms.string(''),
    electronsLabel = cms.string('siStripElectrons'),
    siStripElectronProducer = cms.string('findSiElectrons'),
    trackProducer = cms.string('ctfWithMaterialTracks')
)

findElectronsInSiStrips = cms.Sequence(findSiElectrons*ctfWithMaterialTracks*associateSiElectronsWithTracks)
ctfWithMaterialTracks.src = 'findSiElectrons'

