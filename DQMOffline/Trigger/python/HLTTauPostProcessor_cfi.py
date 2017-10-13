import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.HLTTauDQMOffline_cfi import *

def makeInclusiveAnalyzer(monitorModule):
    m1 = DQMEDHarvester("DQMGenericClient",
        subDirs        = cms.untracked.vstring(monitorModule.DQMBaseFolder.value()+"/"+monitorModule.PathSummaryPlotter.DQMFolder.value()),
        verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
        outputFileName = cms.untracked.string(''),
        resolution     = cms.vstring(),
        efficiency     = cms.vstring(),
        efficiencyProfile = cms.untracked.vstring(
            "PathEfficiency 'Accepted Events per Path;;' helpers/PathTriggerBits helpers/RefEvents"
        ),
    )

    m2 = DQMEDHarvester("HLTTauPostProcessor",
        DQMBaseFolder = cms.untracked.string(monitorModule.DQMBaseFolder.value())
    )

    return (m1, m2)

def makePFTauAnalyzer(monitorModule):
    (m1, m2) = makeInclusiveAnalyzer(monitorModule)
    m1.subDirs.extend([monitorModule.DQMBaseFolder.value()+"/HLT_.*",
                       monitorModule.DQMBaseFolder.value()+"/"+monitorModule.L1Plotter.DQMFolder.value(),
                       monitorModule.DQMBaseFolder.value()+"/.*"])

    def _addEfficiencies(level, quantities, nameFormat, titleObject="#tau", postfix=""):
        if postfix != "":
            postfix = " "+postfix
        for quantity, titleLabel in quantities:
            name = nameFormat % (level, quantity)
            title = "%s %s %s efficiency%s" % (level, titleObject, titleLabel, postfix)
            m1.efficiencyProfile.append("%s '%s' helpers/%sNum helpers/%sDenom" % (name, title, name, name))

    _addEfficiencies("L1", [("Et", "E_{T}"),
                            ("Eta", "#eta"),
                            ("Phi", "#phi")], "%sTau%sEff")
    _addEfficiencies("L1", [("Et", "E_{T}"),
                            ("Eta", "#eta"),
                            ("Phi", "#phi")], "%sIsoTau%sEff")
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sTau%sEff", postfix="(high E_{T})")

    _addEfficiencies("L1", [("Et", "E_{T}")], "%sETM%sEff", "ETM")

    _addEfficiencies("L2", [("Et", "E_{T}"),
                            ("Phi", "#phi")], "%sTrigMET%sEff", "MET")
    _addEfficiencies("tau", [("Et", "E_{T}"),("Eta", "#eta"),("Phi", "#phi")], "%s%sEff", titleObject="")
    _addEfficiencies("muon", [("Et", "E_{T}"),("Eta", "#eta"),("Phi", "#phi")], "%s%sEff", titleObject="")
    _addEfficiencies("electron", [("Et", "E_{T}"),("Eta", "#eta"),("Phi", "#phi")], "%s%sEff", titleObject="")
    _addEfficiencies("met", [("Et", "E_{T}"),("Phi", "#phi")], "%s%sEff", titleObject="")

    for level in ["L2", "L3"]:
        _addEfficiencies(level, [("Et", "p_{T}"),
                                 ("Eta", "#eta"),
                                 ("Phi", "#phi")], "%sTrigTau%sEff")
        _addEfficiencies(level, [("HighEt", "p_{T}")], "%sTrigTau%sEff", postfix="(high p_{T})")
        _addEfficiencies(level, [("Et", "p_{T}"),
                                 ("Eta", "#eta"),
                                 ("Phi", "#phi")], "%sTrigElectron%sEff", "electron")
        _addEfficiencies(level, [("Et", "p_{T}"),
                                 ("Eta", "#eta"),
                                 ("Phi", "#phi")], "%sTrigMuon%sEff", "muon")

    m1.efficiency.append("L3EtaPhiEfficiency 'eta phi eff; #eta; #phi' helpers/L3TrigTauEtaPhiEffNum helpers/L3TrigTauEtaPhiEffDenom")
    m1.efficiency.append("tauEtaPhiEfficiency 'eta phi eff; #eta; #phi' helpers/tauEtaPhiEffNum helpers/tauEtaPhiEffDenom")
    m1.efficiency.append("muonEtaPhiEfficiency 'eta phi eff; #eta; #phi' helpers/muonEtaPhiEffNum helpers/muonEtaPhiEffDenom")
    m1.efficiency.append("electronEtaPhiEfficiency 'eta phi eff; #eta; #phi' helpers/electronEtaPhiEffNum helpers/electronEtaPhiEffDenom")

    return (m1, m2)


(HLTTauPostAnalysisInclusive, HLTTauPostAnalysisInclusive2) = makeInclusiveAnalyzer(hltTauOfflineMonitor_Inclusive)
(HLTTauPostAnalysisPFTaus, HLTTauPostAnalysisPFTaus2) = makePFTauAnalyzer(hltTauOfflineMonitor_PFTaus)
(HLTTauPostAnalysisTP, HLTTauPostAnalysisTP2) = makePFTauAnalyzer(hltTauOfflineMonitor_TagAndProbe)

HLTTauPostSeq = cms.Sequence(
    HLTTauPostAnalysisInclusive+HLTTauPostAnalysisInclusive2+
    HLTTauPostAnalysisPFTaus+HLTTauPostAnalysisPFTaus2+
    HLTTauPostAnalysisTP+HLTTauPostAnalysisTP2
)
