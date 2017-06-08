import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOZeroBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("*ZeroBias*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import siPixelDigis
siPixelDigisForLumiZB = siPixelDigis.clone()
siPixelDigisForLumiZB.InputLabel = cms.InputTag("hltFEDSelectorLumiPixels")

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import siPixelClustersPreSplitting
siPixelClustersForLumiZB = siPixelClustersPreSplitting.clone()
siPixelClustersForLumiZB.src = cms.InputTag("siPixelDigisForLumiZB")

from Calibration.LumiAlCaRecoProducers.alcaPCCProducer_cfi import alcaPCCProducer
alcaPCCProducerZeroBias = alcaPCCProducer.clone()
alcaPCCProducerZeroBias.AlcaPCCProducerParameters.pixelClusterLabel = cms.InputTag("siPixelClustersForLumiZB")
alcaPCCProducerZeroBias.AlcaPCCProducerParameters.trigstring        = cms.untracked.string("alcaPCCZeroBias")

# Sequence #
seqALCARECOAlCaPCCZeroBias = cms.Sequence(ALCARECOZeroBiasHLT + siPixelDigisForLumiZB + siPixelClustersForLumiZB + alcaPCCProducerZeroBias)
