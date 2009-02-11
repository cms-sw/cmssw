# The following comments couldn't be translated into the new config version:

#     bool byName = true

import HLTrigger.HLTfilters.hltHighLevel_cfi

dijetsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = ['HLT_Jet50'],
    throw = False
)


