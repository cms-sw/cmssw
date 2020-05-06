import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.PFMET_cfi import *
from RecoMET.METProducers.pfChMet_cfi import *

##____________________________________________________________________________||
recoPFMETTask = cms.Task(pfMet , particleFlowForChargedMET , pfChMet)
recoPFMET = cms.Sequence(recoPFMETTask)

##____________________________________________________________________________||
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(pfMet,  globalThreshold = 999.)
pp_on_AA_2018.toModify(pfChMet, globalThreshold = 999.)
