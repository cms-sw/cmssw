# producer for alcaisotrk (HCAL isolated tracks)
from TrackingTools.TrackAssociator.default_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *

TrackAssociatorParameterBlock.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("EcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("EcalRecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HBHERecHitCollectionLabel = cms.InputTag("HBHERecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HORecHitCollectionLabel = cms.InputTag("HORecHitCollection")
TrackAssociatorParameterBlock.TrackAssociatorParameters.HFRecHitCollectionLabel = cms.InputTag("HFRecHitCollection")

process.hcalRecoAnalyzer = cms.EDFilter("HcalCorrPFCalculation",
    outputFile = cms.untracked.string('HcalCorrPF.root'),
    eventype = cms.untracked.string('single'),
    mc = cms.untracked.string('yes'),
    sign = cms.untracked.string('*'),
    hcalselector = cms.untracked.string('all'),
#    RespcorrAdd = cms.untracked.bool(True),
#    PFcorrAdd = cms.untracked.bool(True),
    ConeRadiusCm = cms.untracked.double(30.),
    ecalselector = cms.untracked.string('yes')
)

