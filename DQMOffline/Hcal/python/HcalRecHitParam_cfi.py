import FWCore.ParameterSet.Config as cms

hcalRecHitsAnalyzer = cms.EDAnalyzer("HcalRecHitsAnalyzer",
#    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    outputFile                = cms.untracked.string(''),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),

    eventype                  = cms.untracked.string('multi'),
    ecalselector              = cms.untracked.string('yes'),
    hcalselector              = cms.untracked.string('all'),
    useAllHistos              = cms.untracked.bool(False)                                 
)

