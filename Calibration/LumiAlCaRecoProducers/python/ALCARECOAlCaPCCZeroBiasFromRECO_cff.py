import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOZeroBiasFromRECOHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("*ZeroBias*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)

from Calibration.LumiAlCaRecoProducers.alcaPCCProducer_cfi import alcaPCCProducer
alcaPCCProducerZBFromRECO = alcaPCCProducer.clone()
alcaPCCProducerZBFromRECO.AlcaPCCProducerParameters.pixelClusterLabel = cms.InputTag("siPixelClusters")
alcaPCCProducerZBFromRECO.AlcaPCCProducerParameters.trigstring        = cms.untracked.string("alcaPCCZeroBiasFromRECO")


# Sequence #
seqALCARECOAlCaPCCZeroBiasFromRECO = cms.Sequence(ALCARECOZeroBiasFromRECOHLT + alcaPCCProducerZBFromRECO)
