import FWCore.ParameterSet.Config as cms

DiMuonHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/Top/DiMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

DiElectronHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/Top/DiElectron"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

ElecMuonHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/Top/ElecMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

topSingleMuonHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/Top/SemiMuonic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

topSingleElectronHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/Top/SemiElectronic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

SingleTopSingleMuonHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/SingleTop/SingleMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

SingleTopSingleElectronHLTValidationHarvest = cms.EDProducer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLT/TopHLTValidation/SingleTop/SingleElectron"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll",
            "hTriggerMon 'Efficiency per trigger bit' TriggerMonSel TriggerMonAll"
            ),
        resolution = cms.vstring(""),
        )

topHLTriggerValidationHarvest = cms.Sequence(  
        DiMuonHLTValidationHarvest
        *DiElectronHLTValidationHarvest
        *ElecMuonHLTValidationHarvest
        *topSingleMuonHLTValidationHarvest
        *topSingleElectronHLTValidationHarvest
        *SingleTopSingleMuonHLTValidationHarvest
        *SingleTopSingleElectronHLTValidationHarvest	
        )

