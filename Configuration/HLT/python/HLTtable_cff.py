import FWCore.ParameterSet.Config as cms

#
# Choice of HLT tables, for inclusion in [production] cfg files
#
# Pick one and only one from those exported from HLTrigger/Configuration
#
# 1: the online pp menu (GRun)
from HLTrigger.Configuration.HLT_GRun_cff import *
#
# 2: the heavy ions menu (HIon)
# from HLTrigger.Configuration.HLT_HIon_cff import *
