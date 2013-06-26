import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
metPre1HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
metPre2HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
metPre3HLTFilter = copy.deepcopy(hltHighLevel)
metPre1HLTFilter.HLTPaths = ['CandHLT1METPre1']
metPre2HLTFilter.HLTPaths = ['CandHLT1METPre2']
metPre3HLTFilter.HLTPaths = ['CandHLT1METPre3']

