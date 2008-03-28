import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
dijetbalance30HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
dijetbalance60HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
dijetbalance110HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
dijetbalance150HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
dijetbalance200HLTFilter = copy.deepcopy(hltHighLevel)
dijetbalance30HLTFilter.HLTPaths = ['CandHLT2jetAve30']
dijetbalance60HLTFilter.HLTPaths = ['CandHLT2jetAve60']
dijetbalance110HLTFilter.HLTPaths = ['CandHLT2jetAve110']
dijetbalance150HLTFilter.HLTPaths = ['CandHLT2jetAve150']
dijetbalance200HLTFilter.HLTPaths = ['CandHLT2jetAve200']

