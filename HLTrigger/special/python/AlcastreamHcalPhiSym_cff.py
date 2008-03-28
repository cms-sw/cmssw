import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#event selection
l1sHcalPhiSym = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#prescaler
hcalPhiSymPresc = copy.deepcopy(hltPrescaler)
#create the alcareco stream with selected rechits
from HLTrigger.special.HLTHcalPhiSymFilter_cfi import *
l1sPres = cms.Sequence(l1sHcalPhiSym+hcalPhiSymPresc)
doHcalStream = cms.Sequence(cms.SequencePlaceholder("doLocalHcal")+alCaHcalPhiSymStream)
alcaHcalPhiSymSequence = cms.Sequence(cms.SequencePlaceholder("hltBegin")+l1sPres+doHcalStream)
l1sHcalPhiSym.L1SeedsLogicalExpression = 'L1_ZeroBias'
hcalPhiSymPresc.prescaleFactor = 1

