# The following comments couldn't be translated into the new config version:

#     bool byName = true

import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi

gammajetHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_IsoPhoton30_L1I','HLT_IsoPhoton15_L1R'],
    throw = False
)


