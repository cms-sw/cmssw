import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOVertexPixelZeroBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey='VertexPixelZeroBias',
    throw = False # tolerate triggers stated above, but not available
)

# Sequence #
seqALCARECOLumiPixelsMinBias = cms.Sequence(ALCARECOVertexPixelZeroBiasHLT)
