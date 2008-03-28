import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
# $Id: pixelMatchElectronsForHLT.cfi,v 1.9 2008/03/17 13:47:57 ghezzi Exp $
#
pixelMatchElectronsForHLT = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    # needed for CkfTrajectoryBuilder
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    # string TrackLabel = "ctfWithMaterialTracksBarrel"
    TrackProducer = cms.InputTag("ctfWithMaterialTracksBarrel"),
    BSProducer = cms.InputTag("offlineBeamSpot"),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    estimator = cms.string('egammaHLTChi2'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectrons'),
    updator = cms.string('KFUpdator')
)


