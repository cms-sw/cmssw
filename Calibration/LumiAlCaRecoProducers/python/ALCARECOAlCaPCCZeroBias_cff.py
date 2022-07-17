import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOZeroBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    HLTPaths = cms.vstring("AlCa_LumiPixelsCounts_ZeroBias_v*"),
    eventSetupPathsKey='',
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True), # choose logical OR between Triggerbits
    throw = cms.bool(False) # tolerate triggers stated above, but not available
)

from Calibration.LumiAlCaRecoProducers.alcaPCCIntegrator_cfi import alcaPCCIntegrator
alcaPCCIntegratorZeroBias = alcaPCCIntegrator.clone()
alcaPCCIntegratorZeroBias.AlcaPCCIntegratorParameters.ProdInst = "alcaPCCZeroBias"


seqALCARECOAlCaPCCZeroBias = cms.Sequence(ALCARECOZeroBiasHLT + alcaPCCIntegratorZeroBias)
