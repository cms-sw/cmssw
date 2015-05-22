from CMGTools.TTHAnalysis.treeReAnalyzer import *

class EventVars2LSS:
    def __init__(self):
        self.branches = [ "mindr_lep1_jet", "mindr_lep2_jet",
                           "avg_dr_jet",
                           "MT_met_lep1", "MT_met_leplep",
                           "sum_abspz", "sum_sgnpz" ]
    def listBranches(self):
        return self.branches[:]
    def __call__(self,event):
        # make python lists as Collection does not support indexing in slices
        leps = [l for l in Collection(event,"LepGood","nLepGood",4)]
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        (met, metphi)  = event.met_pt, event.met_phi
        njet = len(jets); nlep = len(leps)
        # prepare output
        ret = dict([(name,0.0) for name in self.branches])
        # fill output
        if njet >= 1:
            ret["mindr_lep1_jet"] = min([deltaR(j,leps[0]) for j in jets]) if nlep >= 1 else 0;
            ret["mindr_lep2_jet"] = min([deltaR(j,leps[1]) for j in jets]) if nlep >= 2 else 0;
        if njet >= 2:
            sumdr, ndr = 0, 0
            for i,j in enumerate(jets):
                for i2,j2 in enumerate(jets[i+1:]):
                    ndr   += 1
                    sumdr += deltaR(j,j2)
            ret["avg_dr_jet"] = sumdr/ndr if ndr else 0;
        if nlep > 0:
            ret["MT_met_lep1"] = sqrt( 2*leps[0].pt*met*(1-cos(leps[0].phi-metphi)) )
        if nlep > 1:
            px = leps[0].pt*cos(leps[0].phi) + leps[1].pt*cos(leps[1].phi) + met*cos(metphi) 
            py = leps[0].pt*sin(leps[0].phi) + leps[1].pt*sin(leps[1].phi) + met*sin(metphi) 
            ht = leps[0].pt + leps[1].pt + met
            ret["MT_met_leplep"] = sqrt(max(0,ht**2 - px**2 - py**2))
        if nlep >= 1:
            sumapz, sumspz = 0,0
            for o in leps[:2] + jets:
                pz = o.pt*sinh(o.eta)
                sumspz += pz
                sumapz += abs(pz); 
            ret["sum_abspz"] = sumapz
            ret["sum_sgnpz"] = sumspz
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = EventVars2LSS()
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
