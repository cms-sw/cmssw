import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
heavyChHiggsToTauNuHLTFilter = copy.deepcopy(hltHighLevel)
#CMSSW_2_2_X:
#heavyChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_IsoTau_MET65_Trk20']
#8E29:
#heavyChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_SingleLooseIsoTau20']
#1E31:
heavyChHiggsToTauNuHLTFilter.HLTPaths = ['HLT_SingleIsoTau30_Trk5']
