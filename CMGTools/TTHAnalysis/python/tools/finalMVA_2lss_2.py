from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *

class FinalMVA_2LSS_2:
    def __init__(self):
        self._MVAs = {}
        self._vars= [                 
                MVAVar("mhtJet25 := min(mhtJet25, 300)", func = lambda ev : min(ev.mhtJet25,300)),
                MVAVar("jet1Pt := min(Jet1_pt, 300)", func = lambda ev : min(ev.Jet1_pt,300)),
                MVAVar("jet2Pt := min(Jet2_pt, 300)", func = lambda ev : min(ev.Jet2_pt,300)),
                MVAVar("htJet25 := min(htJet25, 1000)", func = lambda ev : min(ev.htJet25,1000)),
                MVAVar("htJet25ratio1224Lep := (LepGood1_pt*(abs(LepGood1_eta)<1.2) + LepGood2_pt*(abs(LepGood2_eta)<1.2) + Jet1_pt*(abs(Jet1_eta) < 1.2) + Jet2_pt*(abs(Jet2_eta) < 1.2) + Jet3_pt*(abs(Jet3_eta) < 1.2) + Jet4_pt*(abs(Jet4_eta) < 1.2) + Jet5_pt*(abs(Jet5_eta) < 1.2) + Jet6_pt*(abs(Jet6_eta) < 1.2) + Jet7_pt*(abs(Jet7_eta) < 1.2) + Jet8_pt*(abs(Jet8_eta) < 1.2))/ (LepGood1_pt + LepGood2_pt + Jet1_pt*(abs(Jet1_eta) < 2.4) + Jet2_pt*(abs(Jet2_eta) < 2.4) + Jet3_pt*(abs(Jet3_eta) < 2.4) + Jet4_pt*(abs(Jet4_eta) < 2.4) + Jet5_pt*(abs(Jet5_eta) < 2.4) + Jet6_pt*(abs(Jet6_eta) < 2.4) + Jet7_pt*(abs(Jet7_eta) < 2.4) + Jet8_pt*(abs(Jet8_eta) < 2.4))", func = lambda ev : (ev.LepGood1_pt*(abs(ev.LepGood1_eta)<1.2) + ev.LepGood2_pt*(abs(ev.LepGood2_eta)<1.2) + ev.Jet1_pt*(abs(ev.Jet1_eta) < 1.2) + ev.Jet2_pt*(abs(ev.Jet2_eta) < 1.2) + ev.Jet3_pt*(abs(ev.Jet3_eta) < 1.2) + ev.Jet4_pt*(abs(ev.Jet4_eta) < 1.2) + ev.Jet5_pt*(abs(ev.Jet5_eta) < 1.2) + ev.Jet6_pt*(abs(ev.Jet6_eta) < 1.2) + ev.Jet7_pt*(abs(ev.Jet7_eta) < 1.2) + ev.Jet8_pt*(abs(ev.Jet8_eta) < 1.2))/ (ev.LepGood1_pt + ev.LepGood2_pt + ev.Jet1_pt*(abs(ev.Jet1_eta) < 2.4) + ev.Jet2_pt*(abs(ev.Jet2_eta) < 2.4) + ev.Jet3_pt*(abs(ev.Jet3_eta) < 2.4) + ev.Jet4_pt*(abs(ev.Jet4_eta) < 2.4) + ev.Jet5_pt*(abs(ev.Jet5_eta) < 2.4) + ev.Jet6_pt*(abs(ev.Jet6_eta) < 2.4) + ev.Jet7_pt*(abs(ev.Jet7_eta) < 2.4) + ev.Jet8_pt*(abs(ev.Jet8_eta) < 2.4))),
                MVAVar("bestMTopHadPt := min(max(bestMTopHadPt,0),400)", func = lambda ev : min(max(ev.bestMTopHadPt,0),400)),
        ]
        P="/afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/macros/finalMVA/2lss/weights/";
        self._MVAs["finalMVA_2LSS_2"] = MVATool("ee", P+"ttbar_LD.weights.xml", self._vars, rarity=True) 
        
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
            self.sf = FinalMVA_2LSS_2()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

