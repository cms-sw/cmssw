import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECORandomFromRECOHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("*Random*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)

from Calibration.LumiAlCaRecoProducers.alcaPCCProducer_cfi import alcaPCCProducer
alcaPCCProducerRandomFromRECO = alcaPCCProducer.clone()
alcaPCCProducerRandomFromRECO.AlcaPCCProducerParameters.pixelClusterLabel = cms.InputTag("siPixelClusters")
alcaPCCProducerRandomFromRECO.AlcaPCCProducerParameters.trigstring        = cms.untracked.string("alcaPCCRandomFromRECO")


# Sequence #
seqALCARECOAlCaPCCRandomFromRECO = cms.Sequence(ALCARECORandomFromRECOHLT + alcaPCCProducerRandomFromRECO)
