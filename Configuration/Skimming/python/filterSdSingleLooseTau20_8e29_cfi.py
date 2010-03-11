import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterSingleLooseTau20_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterSingleLooseTau20_8e29.HLTPaths = ("HLT_SingleLooseIsoTau20",)
filterSingleLooseTau20_8e29.HLTPathsPrescales  = cms.vuint32(5,)
filterSingleLooseTau20_8e29.HLTOverallPrescale = cms.uint32(1)
filterSingleLooseTau20_8e29.andOr = True
