import FWCore.ParameterSet.Config as cms

from HLTrigger.special.alcastreamHLTEcalRegPi0_cff import *
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#   HLT prescaler
prePi0Ecal = copy.deepcopy(hltPrescaler)
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
l1sEcalPi0 = copy.deepcopy(hltLevel1GTSeed)
doEcalPi0 = cms.Sequence(cms.SequencePlaceholder("doRegionalEgammaEcal")*seqAlcastreamEcalPi0)
hltEcalPi0 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+prePi0Ecal+l1sEcalPi0+doEcalPi0)
prePi0Ecal.prescaleFactor = 1
l1sEcalPi0.L1SeedsLogicalExpression = 'L1_SingleJet15 OR L1_SingleJet20 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100'

