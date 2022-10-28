import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECORandomHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    #HLTPaths = ["AlCa_LumiPixelsCounts_Random_v*"],
    eventSetupPathsKey='AlCaPCCRandom',
    TriggerResultsTag = ("TriggerResults","","HLT"),
    andOr = True, # choose logical OR between Triggerbits
    throw = False # tolerate triggers stated above, but not available
)

from Calibration.LumiAlCaRecoProducers.alcaPCCIntegrator_cfi import alcaPCCIntegrator
alcaPCCIntegratorRandom = alcaPCCIntegrator.clone()
alcaPCCIntegratorRandom.AlcaPCCIntegratorParameters.ProdInst = "alcaPCCRandom"


# Sequence #
seqALCARECOAlCaPCCRandom = cms.Sequence(ALCARECORandomHLT+alcaPCCIntegratorRandom)
