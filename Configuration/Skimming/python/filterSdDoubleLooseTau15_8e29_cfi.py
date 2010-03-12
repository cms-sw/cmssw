import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevelDev_cfi
DoubleLooseTau15_8e29 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
DoubleLooseTau15_8e29.HLTPaths = ("HLT_DoubleLooseIsoTau15",)
DoubleLooseTau15_8e29.HLTPathsPrescales  = cms.vuint32(2,)
DoubleLooseTau15_8e29.HLTOverallPrescale = cms.uint32(1)
DoubleLooseTau15_8e29.andOr = True

filterSdDoubleLooseTau15_8e29 = cms.Path(DoubleLooseTau15_8e29)
