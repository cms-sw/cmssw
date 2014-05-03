import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.PFMET_cfi import *
from RecoMET.METProducers.pfChMet_cfi import *

##____________________________________________________________________________||
recoPFMET = cms.Sequence(pfMet + particleFlowForChargedMET + pfChMet)

##____________________________________________________________________________||
