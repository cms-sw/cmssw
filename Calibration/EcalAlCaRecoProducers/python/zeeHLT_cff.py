# The following comments couldn't be translated into the new config version:

# zeeHLT.cff #########

import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi


report = cms.EDAnalyzer("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)


zeeHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_DoubleIsoEle10_L1I'],
    throw = False
)


