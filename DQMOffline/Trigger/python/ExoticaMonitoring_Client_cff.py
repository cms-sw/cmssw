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

NoBPTXEfficiency = DQMEDHarvester("DQMGenericClient",
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
    ),
)

METplusTrackEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/MET/MET105_IsoTrk50/", "HLT/MET/MET120_IsoTrk50/"),
    verbose = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "effic_met_variable    'MET leg turnON;              CaloMET [GeV]; efficiency'     met_variable_numerator    met_variable_denominator",
        "effic_metPhi          'MET leg efficiency vs phi;   CaloMET phi [rad]; efficiency' metPhi_numerator          metPhi_denominator",
        "effic_muonPt_variable 'Track leg turnON;            Muon p_{T} [GeV]; efficiency'  muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta         'Track leg efficiency vs eta; Muon #eta; efficiency'         muonEta_numerator         muonEta_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS     'MET leg efficiency vs LS; LS; CaloMET leg efficiency' metVsLS_numerator metVsLS_denominator",
        "effic_muonPt_vs_LS 'Track leg efficiency vs LS; LS; Track leg efficiency'  muonPtVsLS_numerator muonPtVsLS_denominator",
    ),

)

exoticaClient = cms.Sequence(
    metEfficiency
    + NoBPTXEfficiency
    + photonEfficiency
    + METplusTrackEfficiency
)

