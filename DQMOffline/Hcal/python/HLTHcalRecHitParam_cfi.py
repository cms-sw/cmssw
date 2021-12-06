import FWCore.ParameterSet.Config as cms

from DQMOffline.Hcal.HcalRecHitParam_cfi import *
hltHCALRecHitsAnalyzer = hcalRecHitsAnalyzer.clone(
      TopFolderName             = 'HLT/HCAL/RecHits',
      HBHERecHitCollectionLabel = 'hltHbhereco',
      HFRecHitCollectionLabel   = 'hltHfreco',
      HORecHitCollectionLabel   = 'hltHoreco',
      EBRecHitCollectionLabel   = 'hltEcalRecHit::EcalRecHitsEB', 
      EERecHitCollectionLabel   = 'hltEcalRecHit::EcalRecHitsEE',
      eventype                  = 'multi', # ?!?
      ecalselector              = 'yes',   # ?!?
      hcalselector              = 'all',   # ?!?
      hep17                     = False
)

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hltHCALRecHitsAnalyzer,
      hep17 = True
)

