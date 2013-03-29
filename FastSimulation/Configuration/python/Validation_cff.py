import FWCore.ParameterSet.Config as cms
from Validation.EventGenerator.BasicGenValidation_cff import *
from FastSimulation.Validation.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *


from FastSimulation.Configuration.CommonInputs_cff import *
if(MixingMode==2):
#    from SimGeneral.TrackingAnalysis.trackingParticlesFastSim_cfi import *
    from FastSimulation.Validation.trackingParticlesFastSim_cfi import *
    mergedtruth.mixLabel = cms.string('mixSimCaloHits')
    mergedtruth.simHitLabel = cms.string('g4SimHits')
    mergedtruthMuon.mixLabel = cms.string('mixSimCaloHits')
    mergedtruthMuon.simHitLabel = cms.string('g4SimHits')


prevalidation = cms.Sequence(globalAssociation+hltassociation_fastsim)
prevalidation_preprod = cms.Sequence(globalAssociation)
prevalidation_prod = cms.Sequence(globalAssociation)
validation = cms.Sequence(basicGenTest_seq+globalValidation+hltvalidation_fastsim) 
validation_preprod = cms.Sequence(basicGenTest_seq+globalValidation_preprod+hltvalidation_preprod_fastsim) 
validation_prod = cms.Sequence(basicGenTest_seq+hltvalidation_prod) 
