import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

dijetEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/JME/Jets/AK4/PF/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_dijetptAvgThr       'Jet Pt turnON;            DiJet_Pt_Avg [GeV]; efficiency'     jetptAvgAThr_numerator    jetptAvgAThr_denominator",
        "effic_dijetphiPrb         'Jet efficiency vs #phi_probe; DiJet_Phi_probe #phi [rad]; efficiency'     jetphiPrb_numerator       jetphiPrb_denominator",
        "effic_dijetetaPrb         'Jet efficiency vs #eta_probe; DiJet_Eta_probe #eta; efficiency'           jetetaPrb_numerator       jetetaPrb_denominator"
    )
  
)

dijetClient = cms.Sequence(
    dijetEfficiency
)
