import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
hltBSoftmuonNjetL1seeds = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
hltBSoftmuonHTL1seeds = copy.deepcopy(hltLevel1GTSeed)
hltBSoftmuonNjetL1seeds.L1SeedsLogicalExpression = 'L1_Mu5_Jet15'
hltBSoftmuonHTL1seeds.L1SeedsLogicalExpression = 'L1_HTT300'

