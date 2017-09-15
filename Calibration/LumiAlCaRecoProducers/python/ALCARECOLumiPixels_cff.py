import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOLumiPixelsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, # choose logical OR between Triggerbits
    eventSetupPathsKey='LumiPixels',
    #HLTPaths = ['AlCa_LumiPixels_Random_*', 'AlCa_LumiPixels_ZeroBias_*'],
    throw = False # tolerate triggers stated above, but not available
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigisForLumi = siPixelDigis.clone()
siPixelDigisForLumi.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

# Modify for if the phase 1 pixel detector is active
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigisForLumi, isUpgrade=cms.untracked.bool(True) )

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
siPixelClustersForLumi = siPixelClustersPreSplitting.clone()
siPixelClustersForLumi.src = cms.InputTag("siPixelDigisForLumi")

# Sequence #
seqALCARECOLumiPixels = cms.Sequence(ALCARECOLumiPixelsHLT + siPixelDigisForLumi + siPixelClustersForLumi)
