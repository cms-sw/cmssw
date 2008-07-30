import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
hltBLifetimeL1seeds = copy.deepcopy(hltLevel1GTSeed)
hltBLifetimeL1seeds.L1SeedsLogicalExpression = 'L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300'

