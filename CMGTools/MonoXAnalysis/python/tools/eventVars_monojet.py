from CMGTools.TTHAnalysis.treeReAnalyzer import *

class EventVarsMonojet:
    def __init__(self):
        self.branches = [ "nMu10V", "nMu20T", "nEle10V", "nEle20T", "nTau18V", "nGamma15V", "nGamma175T",
                          "dphijj", "weight", "jetclean1", "jetclean2", "phmet_pt", "phmet_phi"
                          ]
    def initSampleNormalization(self,sample_nevt):
        self.sample_nevt = sample_nevt        
    def listBranches(self):
        biglist = [ ("nJetClean", "I"), ("nTauClean", "I"), ("nMuSel", "I"),
                    ("iM","I",8,"nMuSel"), ("iJ","I",10,"nJetClean"), ("iT","I",3,"nTauClean"),
                    ("nJetClean30", "I"), ("nTauClean18V", "I") ] 
        for jfloat in "pt eta phi mass btagCSV rawPt".split():
            biglist.append( ("JetClean"+"_"+jfloat,"F",10,"nJetClean") )
        for tfloat in "pt eta phi".split():
            biglist.append( ("TauClean"+"_"+tfloat,"F",3,"nTauClean") )
        self.branches = self.branches + biglist
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
        if tau.pt <= 18 or abs(tau.eta) > 2.3: return False
        return tau.idDecayMode > 0.5 and tau.isoCI3hit < 5.0
    def gammaIdVeto(self,gamma):
        return gamma.pt > 15 and abs(gamma.eta) < 2.5
    def gammaIdTight(self,gamma):
        return gamma.pt > 175 and abs(gamma.eta) < 2.5
    def metNoPh(self,met,photons):
        px = met.Px() + sum([p.p4().Px() for p in photons])
        py = met.Py() + sum([p.p4().Py() for p in photons])
        ret = ROOT.TVector3()
        ret.SetXYZ(px,py,0.)
        return ret
    def __call__(self,event):
        # prepare output
        #ret = dict([(name,0.0) for name in self.branches])
        ret = {}; jetret = {}; tauret = {}
        ret['weight'] = event.xsec * 1000 / self.sample_nevt
        leps = [l for l in Collection(event,"LepGood","nLepGood")]
        ret['nMu10V'] = sum([(abs(l.pdgId)==13 and int(self.lepIdVeto(l))) for l in leps ])
        ret['nMu20T'] = sum([(abs(l.pdgId)==13 and int(self.lepIdTight(l))) for l in leps ])
        ret['nEle10V'] = sum([(abs(l.pdgId)==11 and int(self.lepIdVeto(l))) for l in leps ])
        ret['nEle20T'] = sum([(abs(l.pdgId)==11 and int(self.lepIdTight(l))) for l in leps ])
        taus = [t for t in Collection(event,"TauGood","nTauGood")]
        ret['nTau18V'] = sum([(int(self.tauIdVeto(t))) for t in taus ])
        photons = [p for p in Collection(event,"GammaGood","nGammaGood")]
        ret['nGamma15V'] = sum([(int(self.gammaIdVeto(p))) for p in photons ])
        ret['nGamma175T'] = sum([(int(self.gammaIdTight(p))) for p in photons ])
        # event variables for the monojet analysis
        jets = [j for j in Collection(event,"Jet","nJet")]
        njet = len(jets)
        photonsT = [p for p in photons if self.gammaIdTight(p)]
        #print "check photonsT size is ", len(photonsT), " and nGamma175T = ",ret['nGamma175T']
        (met, metphi)  = event.met_pt, event.met_phi
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)
        phmet = self.metNoPh(metp4,photonsT)
        ret['phmet_pt'] = phmet.Pt()
        ret['phmet_phi'] = phmet.Phi()
        ### muon-jet cleaning
        # Define the loose muons to be cleaned
        ret["iM"] = []
        for il,lep in enumerate(leps):
            if abs(lep.pdgId)==13 and self.lepIdVeto(lep):
                ret["iM"].append(il)
        ret["nMuSel"] = len(ret["iM"])
        # Define cleaned jets 
        ret["iJ"] = []; 
        # 0. mark each jet as clean
        for j in jets: j._clean = True
        # 1. associate to each loose lepton its nearest jet 
        for il in ret["iM"]:
            lep = leps[il]
            best = None; bestdr = 0.4
            for j in jets:
                dr = deltaR(lep,j)
                if dr < bestdr:
                    best = j; bestdr = dr
            if best is not None: best._clean = False
        # 2. compute the jet list
        for ij,j in enumerate(jets):
            if not j._clean: continue
            ret["iJ"].append(ij)
        # 3. sort the jets by pt
        ret["iJ"].sort(key = lambda idx : jets[idx].pt, reverse = True)
        # 4. compute the variables
        for jfloat in "pt eta phi mass btagCSV rawPt".split():
            jetret[jfloat] = []
        dphijj = 999
        ijc = 0
        for idx in ret["iJ"]:
            jet = jets[idx]
            for jfloat in "pt eta phi mass btagCSV rawPt".split():
                jetret[jfloat].append( getattr(jet,jfloat) )
            if   ijc==0: ret['jetclean1'] = jet.chHEF > 0.2 and jet.neHEF < 0.7 and jet.phEF < 0.7
            elif ijc==1: ret['jetclean2'] = jet.neHEF < 0.7 and jet.phEF < 0.9
            if ijc==1: dphijj = deltaPhi(jets[ret["iJ"][0]],jet)
            ijc += 1
        ret["nJetClean"] = len(ret['iJ'])
        # 5. compute the sums 
        ret["nJetClean30"] = 0
        for j in jets:
            if not j._clean: continue
            if j.pt > 30:
                ret["nJetClean30"] += 1
        ret['dphijj'] = dphijj

        ### muon-tau cleaning
        # Define cleaned taus
        ret["iT"] = []; 
        # 0. mark each tau as clean
        for t in taus: t._clean = True
        # 1. associate to each loose lepton its nearest tau 
        for il in ret["iM"]:
            lep = leps[il]
            best = None; bestdr = 0.4
            for t in taus:
                dr = deltaR(lep,t)
                if dr < bestdr:
                    best = t; bestdr = dr
            if best is not None: best._clean = False
        # 2. compute the tau list
        for it,t in enumerate(taus):
            if not t._clean: continue
            ret["iT"].append(it)
        # 3. sort the taus by pt
        ret["iT"].sort(key = lambda idx : taus[idx].pt, reverse = True)
        # 4. compute the variables
        for tfloat in "pt eta phi".split():
            tauret[tfloat] = []
        for idx in ret["iT"]:
            tau = taus[idx]
            for tfloat in "pt eta phi".split():
                tauret[tfloat].append( getattr(tau,tfloat) )
        ret["nTauClean"] = len(ret['iT'])
        # 5. compute the sums 
        ret["nTauClean18V"] = 0
        for t in taus:
            if not t._clean: continue
            if not self.tauIdVeto(t): continue
            ret["nTauClean18V"] += 1
        
        ### return
        fullret = {}
        for k,v in ret.iteritems():
            fullret[k] = v
        for k,v in jetret.iteritems():
            fullret["JetClean_%s" % k] = v
        for k,v in tauret.iteritems():
            fullret["TauClean_%s" % k] = v
        return fullret

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

        
