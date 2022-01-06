import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.pfMet_cfi import *
from RecoMET.METProducers.pfChMet_cfi import *
from CommonTools.PileupAlgos.Puppi_cff import puppiNoLep
from RecoMET.METProducers.pfMetPuppi_cfi import *

##____________________________________________________________________________||
recoPFMETTask = cms.Task(pfMet , particleFlowForChargedMET , pfChMet, puppiNoLep, pfMetPuppi)
recoPFMET = cms.Sequence(recoPFMETTask)

##____________________________________________________________________________||
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(pfMet,  globalThreshold = 999.)
pp_on_AA.toModify(pfChMet, globalThreshold = 999.)
pp_on_AA.toModify(pfMetPuppi,  globalThreshold = 999.)
