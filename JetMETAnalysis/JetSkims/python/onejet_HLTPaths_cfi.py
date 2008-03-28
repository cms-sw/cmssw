import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
onejetpe0HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
onejetpe1HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
onejetpe3HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
onejetpe5HLTFilter = copy.deepcopy(hltHighLevel)
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
onejetpe7HLTFilter = copy.deepcopy(hltHighLevel)
onejetpe0HLTFilter.HLTPaths = ['HLT1jet']
onejetpe1HLTFilter.HLTPaths = ['HLT1jetPE1']
onejetpe3HLTFilter.HLTPaths = ['HLT1jetPE3']
onejetpe5HLTFilter.HLTPaths = ['HLT1jetPE5']
onejetpe7HLTFilter.HLTPaths = ['CandHLT1jetPE7']

