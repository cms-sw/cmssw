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
    hep17                     = cms.untracked.bool(False)
#    useAllHistos              = cms.untracked.bool(False)                                 
)

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hcalRecHitsAnalyzer,
      hep17 = cms.untracked.bool(True)
)
