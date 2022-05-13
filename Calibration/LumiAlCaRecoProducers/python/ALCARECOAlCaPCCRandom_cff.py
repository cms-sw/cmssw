import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECORandomHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("AlCa_LumiPixelsCounts_Random_v*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)


from Calibration.LumiAlCaRecoProducers.alcaPCCIntegrator_cfi import alcaPCCIntegrator
alcaPCCIntegratorRandom = alcaPCCIntegrator.clone()
alcaPCCIntegratorRandom.AlcaPCCIntegratorParameters.inputPccLabel="hltAlcaPixelClusterCounts"
alcaPCCIntegratorRandom.AlcaPCCIntegratorParameters.trigstring    = "alcaPCCEvent"
alcaPCCIntegratorRandom.AlcaPCCIntegratorParameters.ProdInst      = "alcaPCCRandom"


# Sequence #
seqALCARECOAlCaPCCRandom = cms.Sequence(ALCARECORandomHLT+alcaPCCIntegratorRandom)
