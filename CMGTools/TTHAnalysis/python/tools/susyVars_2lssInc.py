from CMGTools.TTHAnalysis.treeReAnalyzer import *

class SusyVars2LSSInc:
    def __init__(self):
        self.branches = [ "iL1", "iL2", "iL1T", "iL2T",
                          "mZ1cut10LL","mZ1cut10TL","minMllAFASTL","minMllAFOSTL","minMllSFOSTL",
                          "nLepGood10T", "nLepGood10noBJet40", "nLepGood10noBJet25", 
                          "nLepGood10relIso04sip10",
                          "nLepGood10relIso04sip10noBJet25",
                          "nBJetMedium40LepRec", "nBJetMedium25LepRec", 
                          "htJet25je08", "htJet25je12", "htJet25j",
                          "htJet40je08", "htJet40je12"  ]
    def listBranches(self):
        return self.branches[:]
    def bestPair(self,leps,cut=lambda lep:True):
        ret = (0,1)
        if len(leps) > 2:
            pairs = []
            for il1 in xrange(len(leps)-1):
                for il2 in xrange(il1+1,len(leps)): 
                    l1 = leps[il1]
                    l2 = leps[il2]
                    if l2.pt < 10: break 
                    if not cut(l1) or not cut(l2): continue
                    if l1.charge == l2.charge:
                        flav = abs(l1.pdgId) + abs(l2.pdgId)
                        ht   = l1.pt + l2.pt
                        pairs.append( (-flav,-ht,il1,il2) )
            if len(pairs):
                pairs.sort()
                ret = (pairs[0][2],pairs[0][3])
        return ret
    def bestZ1WithCut(self,leps,bothcut=lambda lep:True,onecut=lambda lep:True):
          pairs = []
          for il1 in xrange(len(leps)-1):
              for il2 in xrange(il1+1,len(leps)):
                  l1 = leps[il1]
                  l2 = leps[il2]
                  if not bothcut(l1) or not bothcut(l2): continue
                  if not onecut(l1) and not onecut(l2): continue
                  if l1.pdgId == -l2.pdgId:
                      mz = (l1.p4() + l2.p4()).M()
                      diff = abs(mz-91.2)
                      pairs.append( (diff,il1,il2,mz) )
          if len(pairs):
              pairs.sort()
              return pairs[0][3]
          return 0
    def minMllWithCut(self,leps,bothcut=lambda lep:True,onecut=lambda lep:True,paircut=lambda lep1,lep2:True):
            pairs = []
            for il1 in xrange(len(leps)-1):
                for il2 in xrange(il1+1,len(leps)):
                    l1 = leps[il1]
                    l2 = leps[il2]
                    if not bothcut(l1) or not bothcut(l2): continue
                    if not onecut(l1) and not onecut(l2): continue
                    if not paircut(l1,l2): continue
                    mll = (l1.p4() + l2.p4()).M()
                    pairs.append(mll)
            if len(pairs):
                return min(pairs)
            return 0
    def lepId(self,lep):
        if lep.pt <= 10: return False
        if abs(lep.pdgId) == 13:
            if not self.muId(lep): return False
        elif abs(lep.pdgId) == 11:
            if not self.eleId(lep): return False
        return lep.relIso03 < 0.1 and abs(lep.sip3d) < 4
    def muId(self,mu):
        return mu.tightId > 0
    def eleId(self,ele):
        return ele.tightId >= 2 and ele.convVeto and ele.tightCharge > 1 and ele.lostHits == 0
    def __call__(self,event):
        leps = [l for l in Collection(event,"LepGood","nLepGood")]
        iL1iL2_all   = self.bestPair(leps)
        iL1iL2_tight = self.bestPair(leps, cut = lambda lep : self.lepId(lep))
        ret = { 'iL1':iL1iL2_all[0], 'iL2':iL1iL2_all[1],
                'iL1T':iL1iL2_tight[0], 'iL2T':iL1iL2_tight[1], }
        ret['mZ1cut10LL'] = self.bestZ1WithCut(leps, bothcut=lambda l:l.pt>10)
        ret['mZ1cut10TL'] = self.bestZ1WithCut(leps, bothcut=lambda l:l.pt>10, onecut=lambda lep:self.lepId(lep))
        ret['minMllAFASTL'] = self.minMllWithCut(leps, onecut=lambda lep:self.lepId(lep)) 
        ret['minMllAFOSTL'] = self.minMllWithCut(leps, onecut=lambda lep:self.lepId(lep), paircut = lambda l1,l2 : l1.charge!=l2.charge) 
        ret['minMllSFOSTL'] = self.minMllWithCut(leps, onecut=lambda lep:self.lepId(lep), paircut = lambda l1,l2 : l1.pdgId==-l2.pdgId) 
        jets = [j for j in Collection(event,"Jet","nJet")]
        ret['htJet25j']    = sum([j.pt for j in jets if (j.pt > 25)]) # missing in input trees
        ret['htJet25je08'] = sum([j.pt for j in jets if (j.pt > 25 and abs(j.eta) < 0.8)])
        ret['htJet25je12'] = sum([j.pt for j in jets if (j.pt > 25 and abs(j.eta) < 1.2)])
        ret['htJet40je08'] = sum([j.pt for j in jets if (j.pt > 40 and abs(j.eta) < 0.8)])
        ret['htJet40je12'] = sum([j.pt for j in jets if (j.pt > 40 and abs(j.eta) < 1.2)])
        ret['nLepGood10T'] = sum([(self.lepId(l)) for l in leps ])
        ret['nLepGood10noBJet40'] = sum([((l.jetBTagCSV<0.814 or l.pt/l.jetPtRatio<=40) and l.pt > 10) for l in leps ])
        ret['nLepGood10noBJet25'] = sum([((l.jetBTagCSV<0.814 or l.pt/l.jetPtRatio<=25) and l.pt > 10) for l in leps ])
        ret['nLepGood10relIso04sip10'] = sum([(l.relIso03<0.4 and l.sip3d < 10 and l.pt > 10) for l in leps ])
        ret['nLepGood10relIso04sip10noBJet25'] = sum([(l.relIso03<0.4 and l.sip3d < 10 and l.pt > 10 and (l.jetBTagCSV<0.814 or l.pt/l.jetPtRatio<=25)) for l in leps ])
        ret['nBJetMedium40LepRec'] = event.nBJetMedium40 + sum([(l.jetBTagCSV>0.814 and l.pt/l.jetPtRatio>40) for l in leps ])
        ret['nBJetMedium25LepRec'] = event.nBJetMedium25 + sum([(l.jetBTagCSV>0.814 and l.pt/l.jetPtRatio>25) for l in leps ])
        return ret

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("tree")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.sf = SusyVars2LSSInc()
        def analyze(self,ev):
            if ev.nLepGood <= 2: return True
            print "\nrun %6d lumi %4d event %d: leps %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood)
            print self.sf(ev)
            leps = [l for l in Collection(ev,"LepGood","nLepGood")]
            if len(leps) > 2:
                for i,l in enumerate(leps):
                    print "\t%2d  pdgId: %+2d  pT: %6.2f" % (i,l.pdgId,l.pt)
                print ""
    el = EventLoop([ Tester("tester") ])
    el.loop([tree], maxEvents = 50)

        
