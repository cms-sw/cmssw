from CMGTools.TTHAnalysis.treeReAnalyzer import *

class LeptonFakeRateQCDVars:
    def __init__(self,leptonSel,jetSel, jetSort = lambda jet:jet.pt, label=None, isMC=True):
        self.label = "" if (label in ["",None]) else ("_"+label)
        self.leptonSel = leptonSel
        self.jetSel = jetSel
        self.jetSort = jetSort
        self.jetvars = "pt eta phi btagCSV".split()
        if isMC: self.jetvars.append("mcFlavour")
    def listBranches(self):
        label = self.label
        return [ ("nLepGood","I") ] + [ ("LepGood_awayJet%s_%s"%(self.label,var),"F",8,"nLepGood") for var in self.jetvars ]
    def __call__(self,event):
        leps = [l for l in Collection(event,"LepGood","nLepGood")]
        jetsc = [j for j in Collection(event,"Jet","nJet")]
        jetsd = [j for j in Collection(event,"DiscJet","nDiscJet")]
        jetsf = [j for j in Collection(event,"JetFwd","nJetFwd")]
        ret = { "nLepGood" : event.nLepGood }
        for var in self.jetvars:
            ret["LepGood_awayJet%s_%s"%(self.label,var)] = [-99.0] * event.nLepGood
        for il,lep in enumerate(leps):
            if not self.leptonSel(lep): continue
            jets = [ j for j in jetsc+jetsd+jetsf if self.jetSel(j,lep,deltaR(j,lep)) ]
            if len(jets) == 0: continue 
            jet = max(jets, key=self.jetSort)
            #print "lepton pt %6.1f eta %+5.2f phi %+5.2f  matched with jet  pt %6.1f eta %+5.2f phi %+5.2f  " % (
            #    lep.pt, lep.eta, lep.phi, jet.pt, jet.eta, jet.phi )
            for var in self.jetvars:
                ret["LepGood_awayJet%s_%s"%(self.label,var)][il] = getattr(jet,var) 
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("tree")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf1 = LeptonFakeRateQCDVars(
                lambda lep : lep.sip3d < 6 and lep.relIso03 < 0.5,
                lambda jet, lep, dr : jet.pt > (20 if abs(jet.eta)<2.4 else 30) and dr > 0.7)
            self.sf2 = LeptonFakeRateQCDVars(
                lambda lep : lep.sip3d < 6 and lep.relIso03 < 0.5,
                lambda jet, lep, dr : jet.pt > 20 and abs(jet.eta) < 2.4 and dr > 0.7, 
                label="Central")
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf1(ev)
            print self.sf2(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
