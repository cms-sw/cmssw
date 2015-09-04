import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTTauDQMOffline_cfi import *

def makeInclusiveAnalyzer(monitorModule):
    m1 = cms.EDAnalyzer("DQMGenericClient",
        subDirs        = cms.untracked.vstring(monitorModule.DQMBaseFolder.value()+"/"+monitorModule.PathSummaryPlotter.DQMFolder.value()),
        verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
        outputFileName = cms.untracked.string(''),
        resolution     = cms.vstring(),
        efficiency     = cms.vstring(),
        efficiencyProfile = cms.untracked.vstring(
            "PathEfficiency 'Accepted Events per Path;;' helpers/PathTriggerBits helpers/RefEvents"
        ),
    )

    m2 = cms.EDAnalyzer("HLTTauPostProcessor",
        DQMBaseFolder = cms.untracked.string(monitorModule.DQMBaseFolder.value())
    )

    return (m1, m2)

def makePFTauAnalyzer(monitorModule):
    (m1, m2) = makeInclusiveAnalyzer(monitorModule)
    m1.subDirs.extend([monitorModule.DQMBaseFolder.value()+"/HLT_.*",
                       monitorModule.DQMBaseFolder.value()+"/"+monitorModule.L1Plotter.DQMFolder.value()])

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
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sTau%sEff", postfix="(high E_{T})")

    _addEfficiencies("L1", [("Et", "E_{T}"),
                            ("Eta", "#eta"),
                            ("Phi", "#phi")], "%sIsoTau%sEff")
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sIsoTau%sEff", postfix="(high E_{T})")

    _addEfficiencies("L1", [("Et", "E_{T}")], "%sJet%sEff")
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sJet%sEff", "central jet", postfix="(high E_{T})")
    _addEfficiencies("L1", [("Eta", "#eta"),
                            ("Phi", "#phi")], "%sJet%sEff", "central jet", "(E_{T} > %.1f)" % monitorModule.L1Plotter.L1JetMinEt.value())
    _addEfficiencies("L1", [("Et", "E_{T}")], "%sETM%sEff", "ETM")

    _addEfficiencies("L2", [("Et", "E_{T}"),
                            ("Phi", "#phi")], "%sTrigMET%sEff", "MET")

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
    return (m1, m2)


(HLTTauPostAnalysis_Inclusive, HLTTauPostAnalysis_Inclusive2) = makeInclusiveAnalyzer(hltTauOfflineMonitor_Inclusive)
(HLTTauPostAnalysis_PFTaus, HLTTauPostAnalysis_PFTaus2) = makePFTauAnalyzer(hltTauOfflineMonitor_PFTaus)
HLTTauPostSeq = cms.Sequence(
    HLTTauPostAnalysis_Inclusive+HLTTauPostAnalysis_Inclusive2+
    HLTTauPostAnalysis_PFTaus+HLTTauPostAnalysis_PFTaus2
)
