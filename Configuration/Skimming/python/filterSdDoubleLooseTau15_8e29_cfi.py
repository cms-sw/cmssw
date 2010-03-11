import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
filterDoubleLooseTau15_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
filterDoubleLooseTau15_8e29.HLTPaths = ("HLT_DoubleLooseIsoTau15",)
filterDoubleLooseTau15_8e29.HLTPathsPrescales  = cms.vuint32(2,)
filterDoubleLooseTau15_8e29.HLTOverallPrescale = cms.uint32(1)
filterDoubleLooseTau15_8e29.andOr = True
