import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOPCCRandomFilter = copy.deepcopy(hltHighLevel)
ALCARECOPCCRandomFilter.HLTPaths = ['pathALCARECOAlCaPCCRandom']
ALCARECOPCCRandomFilter.throw = True ## dont throw on unknown path names
ALCARECOPCCRandomFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")

from Calibration.LumiAlCaRecoProducers.alcaRawPCCProducer_cfi import *

seqALCARECOPromptCalibProdLumiPCC = cms.Sequence(ALCARECOPCCRandomFilter *
                                                 rawPCCProd)
