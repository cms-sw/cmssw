import FWCore.ParameterSet.Config as cms

from DQMOffline.Hcal.HcalRecHitParam_cfi import *
hltHCALRecHitsAnalyzer = hcalRecHitsAnalyzer.clone()
hltHCALRecHitsAnalyzer.TopFolderName             = cms.string("HLT/HCAL/RecHits")
hltHCALRecHitsAnalyzer.HBHERecHitCollectionLabel = cms.untracked.InputTag("hltHbhereco")
hltHCALRecHitsAnalyzer.HFRecHitCollectionLabel   = cms.untracked.InputTag("hltHfreco")  
hltHCALRecHitsAnalyzer.HORecHitCollectionLabel   = cms.untracked.InputTag("hltHoreco")  
hltHCALRecHitsAnalyzer.EBRecHitCollectionLabel   = cms.InputTag('hltEcalRecHit','EcalRecHitsEB') 
hltHCALRecHitsAnalyzer.EERecHitCollectionLabel   = cms.InputTag('hltEcalRecHit','EcalRecHitsEE')
hltHCALRecHitsAnalyzer.eventype                  = cms.untracked.string('multi') # ?!?
hltHCALRecHitsAnalyzer.ecalselector              = cms.untracked.string('yes')   # ?!?
hltHCALRecHitsAnalyzer.hcalselector              = cms.untracked.string('all')   # ?!?
hltHCALRecHitsAnalyzer.hep17                     = cms.untracked.bool(False)

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hltHCALRecHitsAnalyzer,
      hep17 = cms.untracked.bool(True)
)

