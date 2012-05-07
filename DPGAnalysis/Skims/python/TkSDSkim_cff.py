import FWCore.ParameterSet.Config as cms
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import*
TkSD_Filter = copy.deepcopy(hltHighLevel)
TkSD_Filter.throw = cms.bool(False)
TkSD_Filter.HLTPaths = ["HLT_L1_BscMinBiasOR_BptxPlusORMinus","HLT_L1Tech_BSC_minBias_OR"] 

TkSD_Seq = cms.Sequence(TkSD_Filter)
