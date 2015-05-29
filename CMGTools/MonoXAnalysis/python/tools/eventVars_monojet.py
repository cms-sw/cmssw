from CMGTools.TTHAnalysis.treeReAnalyzer import *

class EventVarsMonojet:
    def __init__(self):
        self.branches = [ "nMu10V", "nMu20T", "nEle10V", "nEle20T", "nTau15V", "nGamma15V", "nGamma175T",
                          "dphijj","jetclean",
                          "weight"]
    def initSampleNormalization(self,sample_nevt):
        self.sample_nevt = sample_nevt        
    def listBranches(self):
        return self.branches[:]
    # physics object multiplicity with the monojet analysis specific selections
    def lepIdVeto(self,lep):
        if lep.pt <= 10: return False
        if abs(lep.pdgId) == 13:
            if abs(lep.eta) > 2.4: return False
            return lep.relIso04 < 0.2
        elif abs(lep.pdgId) == 11:
            if abs(lep.eta) > 2.5: return False
            if lep.relIso03 > (0.164369 if abs(lep.eta)<1.479 else 0.212604): return False
            if lep.dxy > (0.060279 if abs(lep.eta)<1.479 else 0.273097): return False
            if lep.dz > (0.800538 if abs(lep.eta)<1.479 else 0.885860): return False
            if not lep.convVeto: return False
            return lep.lostHits <= (2 if abs(lep.eta)<1.479 else 3)
    def lepIdTight(self,lep):
        if lep.pt <= 20: return False
        if abs(lep.pdgId) == 13:
            return abs(lep.eta) < 2.4 and lep.tightId > 0 and lep.relIso04 < 0.12
        elif abs(lep.pdgId) == 11:
            if lep.dxy > (0.009924 if abs(lep.eta)<1.479 else 0.027261): return False
            if lep.dz > (0.015310 if abs(lep.eta)<1.479 else 0.147154): return False
            return abs(lep.eta) < 2.5 and lep.tightId > 0 and lep.convVeto and lep.lostHits <= 1 and lep.relIso04 < 0.12
    def tauIdVeto(self,tau):
        if tau.pt <= 15 or abs(tau.eta) > 2.3: return False
        return tau.idDecayMode > 0.5 and tau.idCI3hit > 0.5 and tau.idAntiMu > 0.5 and tau.idAntiE > 0.5
    def gammaIdVeto(self,gamma):
        return gamma.pt > 15 and abs(gamma.eta) < 2.5
    def gammaIdTight(self,gamma):
        return gamma.pt > 175 and abs(gamma.eta) < 2.5
    def __call__(self,event):
        # prepare output
        ret = dict([(name,0.0) for name in self.branches])
        ret['weight'] = event.xsec * 1000 / self.sample_nevt
        leps = [l for l in Collection(event,"LepGood","nLepGood")]
        ret['nMu10V'] = sum([(abs(l.pdgId)==13 and int(self.lepIdVeto(l))) for l in leps ])
        ret['nMu20T'] = sum([(abs(l.pdgId)==13 and int(self.lepIdTight(l))) for l in leps ])
        ret['nEle10V'] = sum([(abs(l.pdgId)==11 and int(self.lepIdVeto(l))) for l in leps ])
        ret['nEle20T'] = sum([(abs(l.pdgId)==11 and int(self.lepIdTight(l))) for l in leps ])
        taus = [t for t in Collection(event,"TauGood","nTauGood")]
        ret['nTau15V'] = sum([(int(self.tauIdVeto(t))) for t in taus ])
        photons = [p for p in Collection(event,"GammaGood","nGammaGood")]
        ret['nGamma15V'] = sum([(int(self.gammaIdVeto(p))) for p in photons ])
        ret['nGamma175T'] = sum([(int(self.gammaIdTight(p))) for p in photons ])
        # event variables for the monojet analysis
        jets = [j for j in Collection(event,"Jet","nJet")]
        njet = len(jets)
        ret['dphijj'] = deltaPhi(jets[0],jets[1]) if njet >= 2 else 999 
        if njet >= 1:
            jclean = jets[0].chHEF > 0.2 and jets[0].neHEF < 0.7 and jets[0].phEF < 0.7
            if njet >= 2:
                jclean = jclean and jets[0].neHEF < 0.7 and jets[0].phEF < 0.9
            ret['jetclean'] = jclean
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("tree")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = EventVarsMonojet()
        def analyze(self,ev):
            if ev.metNoMu_pt < 200: return True
            print "\nrun %6d lumi %4d event %d: metNoMu %d" % (ev.run, ev.lumi, ev.evt, ev.metNoMu_pt)
            print self.sf(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
