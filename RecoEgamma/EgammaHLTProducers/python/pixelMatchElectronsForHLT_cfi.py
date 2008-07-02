import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits
# $Id: pixelMatchElectronsForHLT_cfi.py,v 1.2 2008/04/21 03:26:07 rpw Exp $
#
pixelMatchElectronsForHLT = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
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


