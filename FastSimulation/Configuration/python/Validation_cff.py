import FWCore.ParameterSet.Config as cms
from Validation.EventGenerator.BasicGenValidation_cff import *
from FastSimulation.Validation.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *
from DQM.Physics.DQMPhysics_cff import *
from Validation.RecoMET.METRelValForDQM_cff import metPreValidSeq

prevalidation = cms.Sequence(globalAssociation+hltassociation_fastsim+metPreValidSeq)
prevalidation_preprod = cms.Sequence(globalAssociation)
prevalidation_prod = cms.Sequence(globalAssociation)
validation = cms.Sequence(basicGenTest_seq+globalValidation+hltvalidation_fastsim+dqmPhysics) 
validation_preprod = cms.Sequence(basicGenTest_seq+globalValidation_preprod+hltvalidation_preprod_fastsim) 
validation_prod = cms.Sequence(basicGenTest_seq+hltvalidation_prod) 
