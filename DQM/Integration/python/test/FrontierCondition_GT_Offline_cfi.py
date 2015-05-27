import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise

#GlobalTag = gtCustomise(GlobalTag, 'auto:com10_GRun', '')
#customise for 740
GlobalTag = gtCustomise(GlobalTag, 'auto:run2_hlt_GRun', '')
#run2_hlt_50nsGRun
