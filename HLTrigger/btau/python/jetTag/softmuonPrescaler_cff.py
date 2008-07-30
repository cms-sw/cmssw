import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescalerBSoftmuon1jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescalerBSoftmuon2jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescalerBSoftmuon3jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescalerBSoftmuon4jet = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescalerBSoftmuonHT = copy.deepcopy(hltPrescaler)
prescalerBSoftmuon1jet.prescaleFactor = 20

