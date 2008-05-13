import FWCore.ParameterSet.Config as cms

from HLTrigger.special.alcastreamHLTEcalRegPi0_cff import *
import HLTrigger.HLTcore.hltPrescaler_cfi
#   HLT prescaler
prePi0Ecal = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
l1sEcalPi0 = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
doEcalPi0 = cms.Sequence(cms.SequencePlaceholder("doRegionalEgammaEcal")*seqAlcastreamEcalPi0)
hltEcalPi0 = cms.Sequence(cms.SequencePlaceholder("hltBegin")+prePi0Ecal+l1sEcalPi0+doEcalPi0)
prePi0Ecal.prescaleFactor = 1
l1sEcalPi0.L1SeedsLogicalExpression = 'L1GlobalDecision'

