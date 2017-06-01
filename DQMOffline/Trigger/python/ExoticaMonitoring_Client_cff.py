import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
    ),
  
)

NoBPTXEfficiency = DQMEDHarvester("MyHarvester",
    subDirs        = cms.untracked.vstring("HLT/NoBPTX/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetE          'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_numerator          jetE_denominator",
        "effic_jetE_variable 'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_variable_numerator jetE_variable_denominator",
        "effic_jetEta          'Calo jet eta eff;            Jet #eta; Efficiency'     jetEta_numerator          jetEta_denominator",
        "effic_jetPhi          'Calo jet phi eff;            Jet #phi; Efficiency'     jetPhi_numerator          jetPhi_denominator",
        "effic_muonPt          'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_numerator          muonPt_denominator",
        "effic_muonPt_variable 'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta          'Muon eta eff; DisplacedStandAlone Muon #eta; Efficiency'     muonEta_numerator          muonEta_denominator",
        "effic_muonPhi          'Muon phi eff; DisplacedStandAlone Muon #phi; Efficiency'     muonPhi_numerator          muonPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_jetE_vs_LS 'Calo jet energy efficiency vs LS; LS; Jet p_{T} Efficiency' jetEVsLS_numerator jetEVsLS_denominator",
        "effic_jetEta_vs_LS 'Calo jet eta efficiency vs LS; LS; Jet #eta Efficiency' jetEtaVsLS_numerator jetEtaVsLS_denominator",
        "effic_jetPhi_vs_LS 'Calo jet phi efficiency vs LS; LS; Jet #phi Efficiency' jetPhiVsLS_numerator jetPhiVsLS_denominator",
        "effic_muonPt_vs_LS 'Muon pt efficiency vs LS; LS; DSA Muon p_{T} Efficiency' muonPtVsLS_numerator muonPtVsLS_denominator",
        "effic_muonEta_vs_LS 'Muon eta efficiency vs LS; LS; DSA Muon #eta Efficiency' muonEtaVsLS_numerator muonEtaVsLS_denominator",
        "effic_muonPhi_vs_LS 'Muon phi efficiency vs LS; LS; DSA Muon #phi Efficiency' muonPhiVsLS_numerator muonPhiVsLS_denominator",
        "effic_jetE_vs_BX 'Calo jet energy efficiency vs BX; BX; Jet p_{T} Efficiency' jetEVsBX_numerator jetEVsBX_denominator",
        "effic_jetEta_vs_BX 'Calo jet eta efficiency vs BX; BX; Jet #eta Efficiency' jetEtaVsBX_numerator jetEtaVsBX_denominator",
        "effic_jetPhi_vs_BX 'Calo jet phi efficiency vs BX; BX; Jet #phi Efficiency' jetPhiVsBX_numerator jetPhiVsBX_denominator",
        "effic_muonPt_vs_BX 'Muon pt efficiency vs BX; BX; DSA Muon p_{T} Efficiency' muonPtVsBX_numerator muonPtVsBX_denominator",
        "effic_muonEta_vs_BX 'Muon eta efficiency vs BX; BX; DSA Muon #eta Efficiency' muonEtaVsBX_numerator muonEtaVsBX_denominator",
        "effic_muonPhi_vs_BX 'Muon phi efficiency vs BX; BX; DSA Muon #phi Efficiency' muonPhiVsBX_numerator muonPhiVsBX_denominator",
    ), 
)

exoticaClient = cms.Sequence(
    metEfficiency
    + NoBPTXEfficiency
)
