import FWCore.ParameterSet.Config as cms
from Validation.EventGenerator.BasicGenValidation_cff import *
from Validation.Configuration.globalValidation_cff import *
from HLTriggerOffline.Common.HLTValidation_cff import *
from DQM.Physics.DQMPhysics_cff import *
from Validation.RecoMET.METRelValForDQM_cff import metPreValidSeq
from Validation.RecoJets.JetValidation_cff import jetPreValidSeq 

# main sequences
prevalidation = cms.Sequence(globalPrevalidation * hltassociation * metPreValidSeq * jetPreValidSeq)
validation = cms.Sequence(cms.SequencePlaceholder("mix")
                          +genvalid_all
                          *globalValidation
                          *hltvalidation)

# hlt-free versions
prevalidation_noHLT = prevalidation.copy()
prevalidation_noHLT.remove(hltassociation)
validation_noHLT = validation.copy()
validation_noHLT.remove(hltvalidation)
allvalidation_noHLT = cms.Sequence(prevalidation_noHLT+validation_noHLT)
