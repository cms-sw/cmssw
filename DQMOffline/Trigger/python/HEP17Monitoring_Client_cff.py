import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

hep17Efficiency = DQMEDHarvester("DQMGenericClient",
    subDirs    = cms.untracked.vstring("HLT/EGM/HEP17/*"),
    verbose    = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "egamma_passFilter_et_ratioHEP17HEM17              'HEP17/HEM17;E_{T} [GeV];ratio'               egamma_passFilter_et_hep17              egamma_passFilter_et_hem17",
        "egamma_passFilter_etSC_ratioHEP17HEM17            'HEP17/HEM17;E_{T}^{SC} [GeV];ratio'          egamma_passFilter_etSC_hep17            egamma_passFilter_etSC_hem17",
        "egamma_passFilter_energy_ratioHEP17HEM17          'HEP17/HEM17;Energy [GeV];ratio'              egamma_passFilter_energy_hep17          egamma_passFilter_energy_hem17",
        "egamma_passFilter_phi_ratioHEP17HEM17             'HEP17/HEM17;#phi [rad];ratio'                egamma_passFilter_phi_hep17             egamma_passFilter_phi_hem17",
        "egamma_passFilter_hOverE_ratioHEP17HEM17          'HEP17/HEM17;H/E;ratio'                       egamma_passFilter_hOverE_hep17          egamma_passFilter_hOverE_hem17",
        "egamma_passFilter_sigmaIEtaIEta_ratioHEP17HEM17   'HEP17/HEM17;#sigma_{i#etai#eta};ratio'       egamma_passFilter_sigmaIEtaIEta_hep17   egamma_passFilter_sigmaIEtaIEta_hem17",
        "egamma_passFilter_maxr9_ratioHEP17HEM17           'HEP17/HEM17;Max R9;ratio'                    egamma_passFilter_maxr9_hep17           egamma_passFilter_maxr9_hem17",
        "egamma_passFilter_hltIsolEM_ratioHEP17HEM17       'HEP17/HEM17;HLT Iso EM [GeV];ratio'          egamma_passFilter_hltIsolEM_hep17       egamma_passFilter_hltIsolEM_hem17",
        "egamma_passFilter_hltIsolHad_ratioHEP17HEM17      'HEP17/HEM17;HLT Iso Had [GeV];ratio'         egamma_passFilter_hltIsolHad_hep17      egamma_passFilter_hltIsolHad_hem17",
        "egamma_passFilter_hltIsolTrksEle_ratioHEP17HEM17  'HEP17/HEM17;HLT Ele Iso Tracks [GeV];ratio'  egamma_passFilter_hltIsolTrksEle_hep17  egamma_passFilter_hltIsolTrksEle_hem17",
        "egamma_passFilter_HLTenergy_ratioHEP17HEM17       'HEP17/HEM17;HLT Energy [GeV];ratio'          egamma_passFilter_HLTenergy_hep17       egamma_passFilter_HLTenergy_hem17",
        "egamma_passFilter_HLTeta_ratioHEP17HEM17          'HEP17/HEM17;HLT #eta [rad];ratio'            egamma_passFilter_HLTeta_hep17          egamma_passFilter_HLTeta_hem17",
        "egamma_passFilter_HLTphi_ratioHEP17HEM17          'HEP17/HEM17;HLT #phi [rad];ratio'            egamma_passFilter_HLTphi_hep17          egamma_passFilter_HLTphi_hem17",
    ),
    efficiencyProfile = cms.untracked.vstring(
    ),
)
