from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.mvaTool import *

def mt_2(pt1,phi1,pt2,phi2):
    from math import cos, sqrt
    return sqrt(2*pt1*pt2*(1-cos(phi1-phi2)));

class FinalMVA_SUSY_2LSS:
    def __init__(self):
        self._MVAs = {}
        self._vars= [
            MVAVar("mhtJet25 := min(mhtJet25, 400)"),
            MVAVar("met := min(met_pt, 400)"),
            MVAVar("jet1Pt := min(Jet_pt[0], 300)"),
            MVAVar("jet2Pt := min(Jet_pt[1], 300)"),
            MVAVar("htJet25  := min(htJet25, 1000)"),
            MVAVar("htJet40j := min(htJet40j, 1000)"),
            MVAVar("nJet25 := min(nJet25, 8)"),
            MVAVar("lepEta2max := max(abs(LepGood_eta[0]),abs(LepGood_eta[1]))"),
           #MVAVar("lepEta2min := min(abs(LepGood_eta[0]),abs(LepGood_eta[1]))"),
            MVAVar("ptavgEtaJets := (abs(Jet_eta[0])*Jet_pt[0]+abs(Jet_eta[1])*Jet_pt[1])/(Jet_pt[0]+Jet_pt[1])"),
            MVAVar("mtW1   := mt_2(LepGood_pt[0],LepGood_phi[0],met_pt,met_phi)", func = lambda ev: mt_2(ev.LepGood_pt[0],ev.LepGood_phi[0],ev.met_pt,ev.met_phi)),
            MVAVar("mtW2   := mt_2(LepGood_pt[1],LepGood_phi[1],met_pt,met_phi)", func = lambda ev: mt_2(ev.LepGood_pt[1],ev.LepGood_phi[1],ev.met_pt,ev.met_phi)),
            MVAVar("mtWmin := min(mt_2(LepGood_pt[0],LepGood_phi[0],met_pt,met_phi),mt_2(LepGood_pt[1],LepGood_phi[1],met_pt,met_phi))",
                        func = lambda ev : min(mt_2(ev.LepGood_pt[0],ev.LepGood_phi[0],ev.met_pt,ev.met_phi),
                                               mt_2(ev.LepGood_pt[1],ev.LepGood_phi[1],ev.met_pt,ev.met_phi))),
        ]
        P="/afs/cern.ch/work/g/gpetrucc/micro/cmg/CMSSW_7_0_9/src/CMGTools/TTHAnalysis/macros/finalMVA/2lss/weights/";
        self._MVAs["finalMVA_susy_2LSS"] = MVATool("LD", P+"test_LD.weights.xml", self._vars, rarity=True) 
        
    def listBranches(self):
        return self._MVAs.keys()
    def __call__(self,event):
        return dict([ (name, mva(event)) for name, mva in self._MVAs.iteritems()])

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("treeProducerSusyMultilepton")
    #tree.AddFriend("sf/t", argv[2])
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = FinalMVA_SUSY_2LSS()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

