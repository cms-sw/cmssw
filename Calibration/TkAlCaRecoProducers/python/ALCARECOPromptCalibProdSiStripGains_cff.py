import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripGains = copy.deepcopy(hltHighLevel)
ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = ['pathALCARECOSiStripCalMinBias']
ALCARECOCalMinBiasFilterForSiStripGains.throw = True ## dont throw on unknown path names
ALCARECOCalMinBiasFilterForSiStripGains.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


# ------------------------------------------------------------------------------
                     
seqALCARECOPromptCalibProdSiStripGains = cms.Sequence(ALCARECOCalMinBiasFilterForSiStripGains)


