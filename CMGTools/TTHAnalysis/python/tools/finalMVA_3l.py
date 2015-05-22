from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *

class FinalMVA_3L:
    def __init__(self):
        self._MVAs = {}
        self._vars_1_7 = [
                MVAVar("bestMTopHad", func = lambda ev : ev.bestMTopHad),
                MVAVar("nJet25", func = lambda ev : ev.nJet25),
                MVAVar("Jet1_pt", func = lambda ev : ev.Jet1_pt),
                MVAVar("max_Lep_eta := max(max(abs(LepGood1_eta),abs(LepGood2_eta)),abs(LepGood3_eta))", func = lambda ev : max(max(abs(ev.LepGood1_eta),abs(ev.LepGood2_eta)),abs(ev.LepGood3_eta))),
                MVAVar("minDrllAFOS", func = lambda ev : ev.minDrllAFOS),
                MVAVar("htJet25", func = lambda ev : ev.htJet25),
                MVAVar("htJet25ratio1224Lep := (LepGood1_pt*(abs(LepGood1_eta)<1.2) + LepGood2_pt*(abs(LepGood2_eta)<1.2) + LepGood3_pt*(abs(LepGood3_eta)<1.2) + Jet1_pt*(abs(Jet1_eta) < 1.2) + Jet2_pt*(abs(Jet2_eta) < 1.2) + Jet3_pt*(abs(Jet3_eta) < 1.2) + Jet4_pt*(abs(Jet4_eta) < 1.2) + Jet5_pt*(abs(Jet5_eta) < 1.2) + Jet6_pt*(abs(Jet6_eta) < 1.2) + Jet7_pt*(abs(Jet7_eta) < 1.2) + Jet8_pt*(abs(Jet8_eta) < 1.2))/ (LepGood1_pt + LepGood2_pt + LepGood3_pt + Jet1_pt*(abs(Jet1_eta) < 2.4) + Jet2_pt*(abs(Jet2_eta) < 2.4) + Jet3_pt*(abs(Jet3_eta) < 2.4) + Jet4_pt*(abs(Jet4_eta) < 2.4) + Jet5_pt*(abs(Jet5_eta) < 2.4) + Jet6_pt*(abs(Jet6_eta) < 2.4) + Jet7_pt*(abs(Jet7_eta) < 2.4) + Jet8_pt*(abs(Jet8_eta) < 2.4))", func = lambda ev : (ev.LepGood1_pt*(abs(ev.LepGood1_eta)<1.2) + ev.LepGood2_pt*(abs(ev.LepGood2_eta)<1.2) + ev.LepGood3_pt*(abs(ev.LepGood3_eta)<1.2) + ev.Jet1_pt*(abs(ev.Jet1_eta) < 1.2) + ev.Jet2_pt*(abs(ev.Jet2_eta) < 1.2) + ev.Jet3_pt*(abs(ev.Jet3_eta) < 1.2) + ev.Jet4_pt*(abs(ev.Jet4_eta) < 1.2) + ev.Jet5_pt*(abs(ev.Jet5_eta) < 1.2) + ev.Jet6_pt*(abs(ev.Jet6_eta) < 1.2) + ev.Jet7_pt*(abs(ev.Jet7_eta) < 1.2) + ev.Jet8_pt*(abs(ev.Jet8_eta) < 1.2))/ (ev.LepGood1_pt + ev.LepGood2_pt + ev.LepGood3_pt + ev.Jet1_pt*(abs(ev.Jet1_eta) < 2.4) + ev.Jet2_pt*(abs(ev.Jet2_eta) < 2.4) + ev.Jet3_pt*(abs(ev.Jet3_eta) < 2.4) + ev.Jet4_pt*(abs(ev.Jet4_eta) < 2.4) + ev.Jet5_pt*(abs(ev.Jet5_eta) < 2.4) + ev.Jet6_pt*(abs(ev.Jet6_eta) < 2.4) + ev.Jet7_pt*(abs(ev.Jet7_eta) < 2.4) + ev.Jet8_pt*(abs(ev.Jet8_eta) < 2.4)))
        ]
        P="/afs/cern.ch/user/g/gpetrucc/w/TREES_250513_HADD/0_finalmva_3l/weights/"

        self._MVAs["FinalMVA_3L_BDTG"] = MVATool("ee", P+"3l_mix_BDTG.weights.xml", self._vars_1_7) 
        
    def listBranches(self):
        return self._MVAs.keys()
    def __call__(self,event):
        return dict([ (name, mva(event)) for name, mva in self._MVAs.iteritems()])

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    #tree.AddFriend("sf/t", argv[2])
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = FinalMVA_3L()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

