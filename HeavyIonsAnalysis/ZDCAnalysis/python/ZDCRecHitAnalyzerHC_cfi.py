import FWCore.ParameterSet.Config as cms

zdcanalyzer = cms.EDAnalyzer(
   "ZDCRecHitAnalyzerHC",
   ZDCRecHitSource    = cms.InputTag('zdcrecoRun3'),
   ZDCDigiSource    = cms.InputTag('hcalDigis', 'ZDC'),
   AuxZDCRecHitSource    = cms.InputTag('zdcrecoRun3'),
   doZdcRecHits = cms.bool(True),
   doZdcDigis = cms.bool(True),
   doAuxZdcRecHits = cms.bool(False),
   skipRpdRecHits = cms.bool(True), # True: only have ZDC 18 channels in rechit tree
   skipRpdDigis = cms.bool(False), # False: save ZDC 18 channels + RPD 32 channels in digi tree
   doHardcodedRPD = cms.bool(True) # Geometry updated for the RPD are not part of 14_1_X and the GT used for 2024. Always do hard coded RPD for 2024.
)

