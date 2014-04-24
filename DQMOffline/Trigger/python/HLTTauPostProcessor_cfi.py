import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTTauDQMOffline_cfi import *

def makeInclusiveAnalyzer(monitorModule):
    return cms.EDAnalyzer("DQMGenericClient",
        subDirs        = cms.untracked.vstring(monitorModule.DQMBaseFolder.value()+monitorModule.PathSummaryPlotter.DQMFolder.value()),
        verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
        outputFileName = cms.untracked.string(''),
        resolution     = cms.vstring(),
        efficiency     = cms.vstring(),
        efficiencyProfile = cms.untracked.vstring(
            "PathEfficiency 'Accepted Events per Path;;' helpers/PathTriggerBits helpers/RefEvents"
        ),
    )

def makePFTauAnalyzer(monitorModule):
    m = makeInclusiveAnalyzer(monitorModule)
    m.subDirs.extend([monitorModule.DQMBaseFolder.value()+"HLT_.*",
                      monitorModule.DQMBaseFolder.value()+monitorModule.L1Plotter.DQMFolder.value()])

    def _addEfficiencies(level, quantities, nameFormat, titleObject="#tau", postfix=""):
        if postfix != "":
            postfix = " "+postfix
        for quantity, titleLabel in quantities:
            name = nameFormat % (level, quantity)
            title = "%s %s %s efficiency%s" % (level, titleObject, titleLabel, postfix)
            m.efficiencyProfile.append("%s '%s' helpers/%sNum helpers/%sDenom" % (name, title, name, name))


    _addEfficiencies("L1", [("Et", "E_{T}"),
                            ("Eta", "#eta"),
                            ("Phi", "#phi")], "%sTau%sEff")
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sTau%sEff", postfix="(high E_{T})")

    _addEfficiencies("L1", [("Et", "E_{T}")], "%sJet%sEff")
    _addEfficiencies("L1", [("HighEt", "E_{T}")], "%sJet%sEff", "central jet", postfix="(high E_{T})")
    _addEfficiencies("L1", [("Eta", "#eta"),
                            ("Phi", "#phi")], "%sJet%sEff", "central jet", "(E_{T} > %.1f)" % monitorModule.L1Plotter.L1JetMinEt.value())

    for level in ["L2", "L3"]:
        _addEfficiencies(level, [("Et", "p_{T}"),
                                 ("Eta", "#eta"),
                                 ("Phi", "#phi")], "%sTrigTau%sEff")
        _addEfficiencies(level, [("HighEt", "p_{T}")], "%sTrigTau%sEff", postfix="(high p_{T})")

    return m


HLTTauPostAnalysis_Inclusive = makeInclusiveAnalyzer(hltTauOfflineMonitor_Inclusive)
HLTTauPostAnalysis_PFTaus = makePFTauAnalyzer(hltTauOfflineMonitor_PFTaus)
HLTTauPostSeq = cms.Sequence(HLTTauPostAnalysis_Inclusive+HLTTauPostAnalysis_PFTaus)
