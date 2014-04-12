import FWCore.ParameterSet.Config as cms

DiMuonHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/Top/DiMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

DiElectronHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/Top/DiElectron"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

ElecMuonHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/Top/ElecMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

topSingleMuonHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/Top/SemiMuonic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

topSingleElectronHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/Top/SemiElectronic"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

SingleTopSingleMuonHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/SingleTop/SingleMuon"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
            ),
        resolution = cms.vstring(""),
        )

SingleTopSingleElectronHLTValidationHarvest = cms.EDAnalyzer("DQMGenericClient",
        subDirs = cms.untracked.vstring("HLTValidation/SingleTop/SingleElectron"),
        efficiency = cms.vstring(
            "hEffLeptonEta 'Efficiency vs Eta Lepton ' EtaLeptonSel EtaLeptonAll ",
            "hEffLeptonPt 'Efficiency vs Pt Lepton' PtLeptonSel PtLeptonAll ",
            "hEffLastJetEta 'Efficiency vs Eta Last Jet' EtaLastJetSel EtaLastJetAll",
            "hEffLastJetPt 'Efficiency vs Pt Last Jet' PtLastJetSel PtLastJetAll"
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

