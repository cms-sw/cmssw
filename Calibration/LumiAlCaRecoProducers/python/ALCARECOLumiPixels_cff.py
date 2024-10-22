import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOLumiPixelsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey='LumiPixels',
    #HLTPaths = ['AlCa_LumiPixels_Random_*', 'AlCa_LumiPixels_ZeroBias_*'],
    throw = False # tolerate triggers stated above, but not available
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigisForLumi = siPixelDigis.cpu.clone(
    InputLabel = "hltFEDSelectorLumiPixels"
)

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
siPixelClustersForLumi = siPixelClustersPreSplitting.cpu.clone(
    src = "siPixelDigisForLumi"
)

# Sequence #
seqALCARECOLumiPixels = cms.Sequence(ALCARECOLumiPixelsHLT + siPixelDigisForLumi + siPixelClustersForLumi)
