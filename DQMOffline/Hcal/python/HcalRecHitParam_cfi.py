import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hcalRecHitsAnalyzer = DQMEDAnalyzer('HcalRecHitsAnalyzer',
    TopFolderName             = cms.string('HcalRecHitsD/HcalRecHitTask'),
#    outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
    outputFile                = cms.untracked.string(''),

    HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
    HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
    HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
    EBRecHitCollectionLabel   = cms.InputTag("ecalRecHit:EcalRecHitsEB"),                                    
    EERecHitCollectionLabel   = cms.InputTag("ecalRecHit:EcalRecHitsEE"),                                    

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

