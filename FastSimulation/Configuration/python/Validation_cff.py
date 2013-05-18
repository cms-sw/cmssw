import FWCore.ParameterSet.Config as cms
from Validation.EventGenerator.BasicGenValidation_cff import *
from FastSimulation.Validation.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *

mergedtruth.mixLabel = cms.string('mix')
mergedtruth.simHitLabel = cms.string('famosSimHits')
mergedtruth.simHitCollections = cms.PSet(tracker = cms.vstring("famosSimHitsTrackerHits"))
mergedtruthMuon.mixLabel = cms.string('mix')
mergedtruthMuon.simHitLabel = cms.string('famosSimHits')
mergedtruthMuon.simHitCollections = cms.PSet(tracker = cms.vstring("famosSimHitsTrackerHits"))
TrackAssociatorByHits.ROUList = ['famosSimHitsTrackerHits']

prevalidation = cms.Sequence(globalAssociation+hltassociation_fastsim)
prevalidation_preprod = cms.Sequence(globalAssociation)
prevalidation_prod = cms.Sequence(globalAssociation)
validation = cms.Sequence(basicGenTest_seq+globalValidation+hltvalidation_fastsim) 
validation_preprod = cms.Sequence(basicGenTest_seq+globalValidation_preprod+hltvalidation_preprod_fastsim) 
validation_prod = cms.Sequence(basicGenTest_seq+hltvalidation_prod) 
