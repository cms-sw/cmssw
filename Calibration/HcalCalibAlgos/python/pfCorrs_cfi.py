import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
#from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

hcalPFcorrs = cms.EDAnalyzer("HcalCorrPFCalculation",
        TrackAssociatorParameterBlock,
        associationConeSize = cms.double(44.),
        clusterConeSize = cms.double(35.),
        trackIsolationCone = cms.double(63.9),
        neutralIsolationCone = cms.double(46.4),
        ecalCone = cms.double(9.),
        hbheRecHitCollectionTag = cms.InputTag('hbhereco'),
        hfRecHitCollectionTag = cms.InputTag('hfreco'),
        hoRecHitCollectionTag = cms.InputTag('horeco')
#       energyECALmip = cms.double(1.0),
#       RespcorrAdd = cms.untracked.bool(True),
#       PFcorrAdd = cms.untracked.bool(True),
)
