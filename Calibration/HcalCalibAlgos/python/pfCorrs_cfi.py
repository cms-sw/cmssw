import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

#TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("EcalRecHitsEE")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("EcalRecHitsEB")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("HBHERecHitCollection")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("HORecHitCollection")
#TrackAssociatorParameterBlock.TrackAssociatorParameters.HFRecHitCollectionLabel = cms.InputTag("HFRecHitCollection")

hcalRecoAnalyzer = cms.EDAnalyzer("HcalCorrPFCalculation",
   TrackAssociatorParameterBlock,
    outputFile = cms.untracked.string("HcalCorrPF.root"),
    ConeRadiusCm = cms.untracked.double(30.),
     energyECALmip = cms.double(1.0),
#    RespcorrAdd = cms.untracked.bool(True),
#    PFcorrAdd = cms.untracked.bool(True),
)

