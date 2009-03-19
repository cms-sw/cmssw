import FWCore.ParameterSet.Config as cms

#
# Choice of HLT tables, for inclusion in [production] cfg files
#
# Pick one and only one from those exported from HLTrigger/Configuration
#
# 1: the fat old 160-paths HLT table used up to 22X
# from HLTrigger.Configuration.HLT_2E30_cff import *
#
# 2: the new lean low-lumi table (8e29)
from HLTrigger.Configuration.HLT_8E29_cff import *
#
# 3: the new lean high-lumi table (1e31)
# from HLTrigger.Configuration.HLT_1E31_cff import *
