import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
# $Id: pixelMatchElectronsForHLT_cfi.py,v 1.3 2008/07/02 16:12:04 ghezzi Exp $
#
pixelMatchElectronsForHLT = cms.EDProducer("EgammaHLTPixelMatchElectronProducers",
    # needed for CkfTrajectoryBuilder
    # propagatorAlong = cms.string('PropagatorWithMaterial'),
    # nested parameter set for TransientInitialStateEstimator
    # TransientInitialStateEstimatorParameters = cms.PSet(
      #  propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
      #  propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    #),
    # string TrackLabel = "ctfWithMaterialTracksBarrel"
    TrackProducer = cms.InputTag("ctfWithMaterialTracksBarrel"),
    BSProducer = cms.InputTag("offlineBeamSpot")
    # propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    # estimator = cms.string('egammaHLTChi2'),
    # TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectrons'),
    # updator = cms.string('KFUpdator')
)


