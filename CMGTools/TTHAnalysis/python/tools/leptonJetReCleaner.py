from CMGTools.TTHAnalysis.treeReAnalyzer import *

class LeptonJetReCleaner:
    def __init__(self,label,looseLeptonSel,tightLeptonSel,cleanJet,isMC=True):
        self.label = "" if (label in ["",None]) else ("_"+label)
        self.looseLeptonSel = looseLeptonSel
        self.tightLeptonSel = tightLeptonSel
        self.cleanJet = cleanJet
        self.isMC = isMC
    def listBranches(self):
        label = self.label
        biglist = [ ("nLepSel"+label, "I"), ("nLepTight"+label, "I"), ("nJetSel"+label, "I"), 
                 ("iL"+label,"I",20,"nLepSel"+label), ("iLT"+label,"I",20,"nLepTight"+label), 
                 ("iJ"+label,"I",20,"nJetSel"+label), # index >= 0 if in Jet; -1-index (<0) if in DiscJet
                 ("nLepGood10"+label, "I"), ("nLepGood10T"+label, "I"),
                 ("nJet40"+label, "I"), "htJet40j"+label, ("nBJetLoose40"+label, "I"), ("nBJetMedium40"+label, "I"),
                 ("nJet25"+label, "I"), "htJet25j"+label, ("nBJetLoose25"+label, "I"), ("nBJetMedium25"+label, "I"), 
                 ("iL1"+label, "I"), ("iL2"+label, "I"), 
                 ("iL1T"+label, "I"), ("iL2T"+label, "I"), 
                 ("iL1p"+label, "I"), ("iL2p"+label, "I"), 
                 ("iL1Tp"+label, "I"), ("iL2Tp"+label, "I"), 
                 "mZ1cut10TL"+label, "minMllAFASTL"+label,"minMllAFOSTL"+label,"minMllSFOSTL"+label,
                 "mZ1"+label, "minMllAFAS"+label,"minMllAFOS"+label,"minMllSFOS"+label,
                ]
        for jfloat in "pt eta phi mass btagCSV rawPt".split():
            biglist.append( ("JetSel"+label+"_"+jfloat,"F",20,"nJetSel"+label) )
        if self.isMC:
            biglist.append( ("JetSel"+label+"_mcPt",     "F",20,"nJetSel"+label) )
            biglist.append( ("JetSel"+label+"_mcFlavour","I",20,"nJetSel"+label) )
            biglist.append( ("JetSel"+label+"_mcMatchId","I",20,"nJetSel"+label) )
        return biglist
    def __call__(self,event):
        leps = [l for l in Collection(event,"LepGood","nLepGood")]
        jetsc = [j for j in Collection(event,"Jet","nJet")]
        jetsd = [j for j in Collection(event,"DiscJet","nDiscJet")]
        ret = {}; jetret = {}
        #
        ### Define loose leptons
        ret["iL"] = []; ret["nLepGood10"] = 0
        for il,lep in enumerate(leps):
            if self.looseLeptonSel(lep):
                ret["iL"].append(il)
                if lep.pt > 10: ret["nLepGood10"] += 1
        ret["nLepSel"] = len(ret["iL"])
        #
        ### Define tight leptons
        ret["iLT"] = []; ret["nLepGood10T"] = 0
        for il in ret["iL"]:
            lep = leps[il]
            if self.tightLeptonSel(lep):
                ret["iLT"].append(il)
                if lep.pt > 10: ret["nLepGood10T"] += 1
        ret["nLepTight"] = len(ret["iLT"])
        #
        ### Define jets
        ret["iJ"] = []
        # 0. mark each jet as clean
        for j in jetsc+jetsd: j._clean = True
        # 1. associate to each loose lepton its nearest jet 
        for il in ret["iL"]:
            lep = leps[il]
            best = None; bestdr = 0.4
            for j in jetsc+jetsd:
                dr = deltaR(lep,j)
                if dr < bestdr:
                    best = j; bestdr = dr
            if best is not None and self.cleanJet(lep,best,bestdr):
                best._clean = False
        # 2. compute the jet list
        for ijc,j in enumerate(jetsc):
            if not j._clean: continue
            ret["iJ"].append(ijc)
        for ijd,j in enumerate(jetsd):
            if not j._clean: continue
            ret["iJ"].append(-1-ijd)
        # 3. sort the jets by pt
        ret["iJ"].sort(key = lambda idx : jetsc[idx].pt if idx >= 0 else jetsd[-1-idx].pt, reverse = True)
        # 4. compute the variables
        for jfloat in "pt eta phi mass btagCSV rawPt".split():
            jetret[jfloat] = []
        if self.isMC:
            for jmc in "mcPt mcFlavour mcMatchId".split():
                jetret[jmc] = []
        for idx in ret["iJ"]:
            jet = jetsc[idx] if idx >= 0 else jetsd[-1-idx]
            for jfloat in "pt eta phi mass btagCSV rawPt".split():
                jetret[jfloat].append( getattr(jet,jfloat) )
            if self.isMC:
                for jmc in "mcPt mcFlavour mcMatchId".split():
                    jetret[jmc].append( getattr(jet,jmc) )
        # 5. compute the sums
        ret["nJet25"] = 0; ret["htJet25j"] = 0; ret["nBJetLoose25"] = 0; ret["nBJetMedium25"] = 0
        ret["nJet40"] = 0; ret["htJet40j"] = 0; ret["nBJetLoose40"] = 0; ret["nBJetMedium40"] = 0
        for j in jetsc+jetsd:
            if not j._clean: continue
            if j.pt > 25:
                ret["nJet25"] += 1; ret["htJet25j"] += j.pt; 
                if j.btagCSV>0.423: ret["nBJetLoose25"] += 1
                if j.btagCSV>0.814: ret["nBJetMedium25"] += 1
            if j.pt > 40:
                ret["nJet40"] += 1; ret["htJet40j"] += j.pt; 
                if j.btagCSV>0.423: ret["nBJetLoose40"] += 1
                if j.btagCSV>0.814: ret["nBJetMedium40"] += 1
        #
        ### 2lss specific things
        lepsl = [ leps[il] for il in ret["iL"]  ]
        lepst = [ leps[il] for il in ret["iLT"] ]
        ret['mZ1'] = self.bestZ1TL(lepsl, lepsl)
        ret['mZ1cut10TL'] = self.bestZ1TL(lepsl, lepst, cut=lambda l:l.pt>10)
        ret['minMllAFAS'] = self.minMllTL(lepsl, lepsl) 
        ret['minMllAFOS'] = self.minMllTL(lepsl, lepsl, paircut = lambda l1,l2 : l1.charge !=  l2.charge) 
        ret['minMllSFOS'] = self.minMllTL(lepsl, lepsl, paircut = lambda l1,l2 : l1.pdgId  == -l2.pdgId) 
        ret['minMllAFASTL'] = self.minMllTL(lepsl, lepst) 
        ret['minMllAFOSTL'] = self.minMllTL(lepsl, lepst, paircut = lambda l1,l2 : l1.charge !=  l2.charge) 
        ret['minMllSFOSTL'] = self.minMllTL(lepsl, lepst, paircut = lambda l1,l2 : l1.pdgId  == -l2.pdgId) 
        for (name,lepcoll,byflav) in ("",lepsl,True),("p",lepsl,False),("T",lepst,True),("Tp",lepst,False):
            iL1iL2 = self.bestSSPair(lepcoll, byflav, cut = lambda lep : lep.pt > 10)
            ret["iL1"+name] = iL1iL2[0]
            ret["iL2"+name] = iL1iL2[1]
        #
        ### attach labels and return
        fullret = {}
        for k,v in ret.iteritems(): 
            fullret[k+self.label] = v
        for k,v in jetret.iteritems(): 
            fullret["JetSel%s_%s" % (self.label,k)] = v
        return fullret
    def bestZ1TL(self,lepsl,lepst,cut=lambda lep:True):
          pairs = []
          for l1 in lepst:
            if not cut(l1): continue
            for l2 in lepsl:
                if not cut(l2): continue
                if l1.pdgId == -l2.pdgId:
                   mz = (l1.p4() + l2.p4()).M()
                   diff = abs(mz-91.2)
                   pairs.append( (diff,mz) )
          if len(pairs):
              pairs.sort()
              return pairs[0][1]
          return 0.
    def minMllTL(self, lepsl, lepst, bothcut=lambda lep:True, onecut=lambda lep:True, paircut=lambda lep1,lep2:True):
            pairs = []
            for l1 in lepst:
                if not bothcut(l1): continue
                for l2 in lepsl:
                    if l2 == l1 or not bothcut(l2): continue
                    if not onecut(l1) and not onecut(l2): continue
                    if not paircut(l1,l2): continue
                    mll = (l1.p4() + l2.p4()).M()
                    pairs.append(mll)
            if len(pairs):
                return min(pairs)
            return -1
    def bestSSPair(self,leps,byflav,cut=lambda lep:True):
        ret = (0,1)
        if len(leps) > 2:
            pairs = []
            for il1 in xrange(len(leps)-1):
                for il2 in xrange(il1+1,len(leps)): 
                    l1 = leps[il1]
                    l2 = leps[il2]
                    if not cut(l1) or not cut(l2): continue
                    if l1.charge == l2.charge:
                        flav = abs(l1.pdgId) + abs(l2.pdgId) if byflav else 0
                        ht   = l1.pt + l2.pt
                        pairs.append( (-flav,-ht,il1,il2) )
            if len(pairs):
                pairs.sort()
                ret = (pairs[0][2],pairs[0][3])
        return ret


def _tthlep_lepId(lep):
        #if lep.pt <= 10: return False
        if abs(lep.pdgId) == 13:
            if lep.pt <= 5: return False
            return True #lep.mediumMuonId > 0
        elif abs(lep.pdgId) == 11:
            if lep.pt <= 7: return False
            if not (lep.convVeto and lep.lostHits == 0): 
                return False
            return True #lep.mvaIdPhys14 > 0.73+(0.57-0.74)*(abs(lep.eta)>0.8)+(0.05-0.57)*(abs(lep.eta)>1.479)
        return False

def _susy2lss_lepId_CB(lep):
        if lep.pt <= 10: return False
        if abs(lep.pdgId) == 13:
            return lep.mediumMuonId > 0
        elif abs(lep.pdgId) == 11:
            if not (lep.convVeto and lep.tightCharge > 1 and lep.lostHits == 0): 
                return False
            return lep.mvaIdPhys14 > 0.73+(0.57-0.74)*(abs(lep.eta)>0.8)+(0.05-0.57)*(abs(lep.eta)>1.479)
        return False

def _susy2lss_lepId_CBOld(lep):
        if lep.pt <= 10: return False
        if abs(lep.pdgId) == 13:
            return lep.tightId > 0
        elif abs(lep.pdgId) == 11:
            return lep.tightId >= 2 and lep.convVeto and lep.tightCharge > 1 and lep.lostHits == 0
        return False

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("tree")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf1 = LeptonJetReCleaner("Old", 
                lambda lep : lep.relIso03 < 0.5, 
                lambda lep : lep.relIso03 < 0.1 and lep.sip3d < 4 and _susy2lss_lepId_CB(lep),
                cleanJet = lambda lep,jet,dr : (lep.pt > 10 and dr < 0.4))
            self.sf2 = LeptonJetReCleaner("PtRel", 
                lambda lep : lep.relIso03 < 0.4 or lep.jetPtRel > 5, 
                lambda lep : (lep.relIso03 < 0.1 or lep.jetPtRel > 14) and lep.sip3d < 4 and _susy2lss_lepId_CB(lep),
                cleanJet = lambda lep,jet,dr : (lep.pt > 10 and dr < 0.4))
            self.sf3 = LeptonJetReCleaner("MiniIso", 
                lambda lep : lep.miniRelIso < 0.4, 
                lambda lep : lep.miniRelIso < 0.05 and lep.sip3d < 4 and _susy2lss_lepId_CB(lep),
                cleanJet = lambda lep,jet,dr : (lep.pt > 10 and dr < 0.4))
            self.sf4 = LeptonJetReCleaner("PtRelJC", 
                lambda lep : lep.relIso03 < 0.4 or lep.jetPtRel > 5, 
                lambda lep : (lep.relIso03 < 0.1 or lep.jetPtRel > 14) and lep.sip3d < 4 and _susy2lss_lepId_CB(lep),
                cleanJet = lambda lep,jet,dr : (lep.pt > 10 and dr < 0.4 and not (lep.jetPtRel > 5 and lep.pt*(1/lep.jetPtRatio-1) > 25)))
            self.sf5 = LeptonJetReCleaner("MiniIsoJC", 
                lambda lep : lep.miniRelIso < 0.4, 
                lambda lep : lep.miniRelIso < 0.05 and lep.sip3d < 4 and _susy2lss_lepId_CB(lep),
                cleanJet = lambda lep,jet,dr : (lep.pt > 10 and dr < 0.4 and not (lep.jetDR > 0.5*10/min(50,max(lep.pt,200)) and lep.pt*(1/lep.jetPtRatio-1) > 25)))
        def analyze(self,ev):
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf1(ev)
            print self.sf2(ev)
            print self.sf3(ev)
            print self.sf4(ev)
            print self.sf5(ev)
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
