import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECORandomHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("*Random*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigisForLumiR = siPixelDigis.clone()
siPixelDigisForLumiR.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
siPixelClustersForLumiR = siPixelClustersPreSplitting.clone()
siPixelClustersForLumiR.src = cms.InputTag("siPixelDigisForLumiR")

from Calibration.LumiAlCaRecoProducers.alcaPCCProducer_cfi import alcaPCCProducer
alcaPCCProducerRandom = alcaPCCProducer.clone()
alcaPCCProducerRandom.AlcaPCCProducerParameters.pixelClusterLabel = cms.InputTag("siPixelClustersForLumiR")
alcaPCCProducerRandom.AlcaPCCProducerParameters.trigstring        = cms.untracked.string("alcaPCCRandom")

# Sequence #
seqALCARECOAlCaPCCRandom = cms.Sequence(ALCARECORandomHLT + siPixelDigisForLumiR + siPixelClustersForLumiR + alcaPCCProducerRandom)
