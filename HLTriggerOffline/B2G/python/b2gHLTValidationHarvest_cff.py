import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


b2gSingleMuonHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/SemiMuonic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

b2gDoubleLeptonEleMuHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/EleMu"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

b2gDoubleElectronHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/DoubleEle"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

b2gSingleElectronHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/SemiElectronic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

b2gSingleJetHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/SingleJet"),
        efficiency = cms.vstring(
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

b2gDiJetHLTValidationHarvest = DQMEDHarvester("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/B2GHLTValidation/B2G/DiJet"),
        efficiency = cms.vstring(
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )


b2gHLTriggerValidationHarvest = cms.Sequence(  
    b2gSingleMuonHLTValidationHarvest
    *b2gSingleElectronHLTValidationHarvest
    *b2gSingleJetHLTValidationHarvest
    *b2gDiJetHLTValidationHarvest
    *b2gDoubleElectronHLTValidationHarvest
    *b2gDoubleLeptonEleMuHLTValidationHarvest
    )

