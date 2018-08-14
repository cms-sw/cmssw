import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hep17Efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/EgOffline/HEP17/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "ratio_et          'HEP17/HEM17;            Et [GeV]; ratio'     egamma_passFilter_et_hep17        egamma_passFilter_et_hem17",
        "ratio_hOverE          'HEP17/HEM17;        hOverE; ratio'     egamma_passFilter_hOverE_hep17        egamma_passFilter_hOverE_hem17",
       # "ratio_isolEM          'HEP17/HEM17;        isolEM [GeV]; ratio'     egamma_passFilter_isolEM_hep17        egamma_passFilter_isolEM_hem17",
       # "ratio_isolHad         'HEP17/HEM17;        isolHad [GeV]; ratio'     egamma_passFilter_isolHad_hep17        egamma_passFilter_isolHad_hem17",
        "ratio_sigmaIEtaIEta   'HEP17/HEM17;       sigmaIEtaIEta; ratio'     egamma_passFilter_sigmaIEtaIEta_hep17        egamma_passFilter_sigmaIEtaIEta_hem17",
        "ratio_energy          'HEP17/HEM17;        energy [GeV]; ratio'     egamma_passFilter_energy_hep17        egamma_passFilter_energy_hem17",
        "ratio_r9              'HEP17/HEM17;            r9 ;ratio'     egamma_passFilter_maxr9_hep17        egamma_passFilter_maxr9_hem17",

        "ratio_phi              'HEP17/HEM17;      phi [rad] ;ratio'     egamma_passFilter_phi_hep17        egamma_passFilter_phi_hem17",
       # "ratio_isolNrTrks          'HEP17/HEM17;    isolNrTrks [GeV]; ratio'     egamma_passFilter_isolNrTrks_hep17        egamma_passFilter_isolNrTrks_hem17",
        "ratio_etSC          'HEP17/HEM17;          etSC [GeV]; ratio'     egamma_passFilter_etSC_hep17        egamma_passFilter_etSC_hem17",
        "ratio_HLTenergy     HEP17/HEM17;            HLTenergy [GeV]; ratio'     egamma_passFilter_HLTenergy_hep17    egamma_passFilter_HLTenergy_hem17",
        "ratio_HLTphi              'HEP17/HEM17;      HLTphi [rad]; ratio'     egamma_passFilter_HLTphi_hep17        egamma_passFilter_HLTphi_hem17",
    ),
    efficiencyProfile = cms.untracked.vstring(
    ),
  
)
