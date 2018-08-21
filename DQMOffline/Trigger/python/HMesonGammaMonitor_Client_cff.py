import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hmesongammaEfficiency = DQMEDHarvester("DQMGenericClient",
#    subDirs        = cms.untracked.vstring("HLT/Higgs/HMesonGamma/*"),
    subDirs        = cms.untracked.vstring("HLT/HIG/HMesonGamma/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
       
        "effic_gammapt          'gamma pT turnON;            PFGamma pT [GeV]; efficiency'     gammapt_numerator          gammapt_denominator",
        "effic_mesonpt          'meson pT turnON;   PFMeson pT [GeV]; efficiency'     mesonpt_numerator       mesonpt_denominator",
        "effic_gammaeta          'gamma #eta turnON;   gamma #eta; efficiency'     gammaeta_numerator       gammaeta_denominator",
        "effic_mesoneta          'meson #eta turnON;   meson #eta; efficiency'     mesoneta_numerator       mesoneta_denominator",
       
        
    ),
    efficiencyProfile = cms.untracked.vstring(
        
        "effic_gammaetaVsLS 'Gamma #eta efficiency vs LS; LS; gamma #eta efficiency' gammaetaVsLS_numerator gammaetaVsLS_denominator",

    ),
  
)


hmesongammaClient = cms.Sequence(
    hmesongammaEfficiency
)
