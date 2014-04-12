import FWCore.ParameterSet.Config as cms
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import*
TkSDFilter = copy.deepcopy(hltHighLevel)
TkSDFilter.throw = cms.bool(False)
TkSDFilter.HLTPaths = ["HLT_L1_BscMinBiasOR_BptxPlusORMinus","HLT_L1Tech_BSC_minBias_OR"] 

TkSD_Seq = cms.Sequence(TkSDFilter)
