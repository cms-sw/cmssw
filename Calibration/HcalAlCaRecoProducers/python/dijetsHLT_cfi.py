# The following comments couldn't be translated into the new config version:

#     bool byName = true

import HLTrigger.HLTfilters.hltHighLevel_cfi

dijetsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_DiJetAve15','HLT_DiJetAve30','HLT_DiJetAve50','HLT_Jet30','HLT_Jet50'],
    throw = False
)


