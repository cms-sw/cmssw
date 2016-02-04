import FWCore.ParameterSet.Config as cms
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import*
HighPU_Filter = copy.deepcopy(hltHighLevel)
HighPU_Filter.throw = cms.bool(False)
HighPU_Filter.HLTPaths = ["HLT_60Jet10_v*","HLT_70Jet10_v*","HLT_70Jet13_v*"]

HighPU_Seq = cms.Sequence(HighPU_Filter)
