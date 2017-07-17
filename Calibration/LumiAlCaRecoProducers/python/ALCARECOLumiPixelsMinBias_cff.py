import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOLumiPixelsMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey='LumiPixelsMinBias',
    throw = False # tolerate triggers stated above, but not available
)

# Sequence #
seqALCARECOLumiPixelsMinBias = cms.Sequence(ALCARECOLumiPixelsMinBiasHLT)
