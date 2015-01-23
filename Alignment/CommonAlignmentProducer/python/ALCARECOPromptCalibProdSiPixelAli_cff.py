import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOTkAlMinBiasFilterForSiPixelAli = copy.deepcopy(hltHighLevel)
ALCARECOTkAlMinBiasFilterForSiPixelAli.HLTPaths = ['pathALCARECOTkAlMinBias']
ALCARECOTkAlMinBiasFilterForSiPixelAli.throw = True ## dont throw on unknown path names
ALCARECOTkAlMinBiasFilterForSiPixelAli.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")


seqALCARECOPromptCalibProdSiPixelAli = cms.Sequence(ALCARECOTkAlMinBiasFilterForSiPixelAli)
