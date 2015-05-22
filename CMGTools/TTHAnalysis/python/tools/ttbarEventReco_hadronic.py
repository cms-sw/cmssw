from CMGTools.TTHAnalysis.treeReAnalyzer import *
from CMGTools.TTHAnalysis.tools.ttbarEventReco_2lss import jetEtResolution,TTLikelihood,bestByX,BestBySingleFunc,BestByCascade

class TTHadEventReco_MC:
    def __init__(self):
        self.branches = [ "n_b", "n_w_j", "n_w_jj", "n_t_jjb", "n_other_b",
                          "i_b1", "i_b2", "i_w1_j1", "i_w1_j2", "i_w2_j1", "i_w2_j2" ]
        # t    -> b1 w+ -> b1 w1j1 w1j2
        # tbar -> b2 w- -> b2 w2j1 w2j2
    def listBranches(self):
        return [ "mc_"+x for x in self.branches ]
    def __call__(self,event):
        ## prepare output container
        ret  = dict([(name,0.0) for name in self.branches])
        ## return if not on MC
        ret0 = dict([("mc_"+name,0.0) for name in self.branches])
        if event.run != 1:   return ret0
        if event.nJet25 < 6: return ret0
        # make python lists as Collection does not support indexing in slices
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
        if len(bjets) < 2: bjets = [ jets[0], jets[1] ]
        njet = len(jets); nb = len(bjets);
        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuarks",2)]
        tquarks = [j for j in Collection(event,"GenTop")]
        if len(tquarks) != 2: return ret0
        lquarks = [j for j in Collection(event,"GenQuark")]
        # --------------------------------------------------------------------------
        # get b-jets
        ret["n_b"] = 0; ret["i_b1"] = -1; ret["i_b2"] = -1 
        for ib, bj in enumerate(bjets):
            bj.quark = None
            if bj.mcMatchId == 6 and bj.mcMatchFlav == 5: 
                #print "jet %2d at pt %6.1f eta %+.3f phi %+.3f (%d, %d, %+2d)" % (ib,bj.pt,bj.eta,bj.phi,bj.mcMatchId,bj.mcMatchFlav,bj.mcFlavour)
                #for iq,q in enumerate(bquarks):
                #    print "   quark %2d at pt %6.1f eta %+.3f phi %+.3f --> dr = %.3f" % (iq,q.pt,q.eta,q.phi,deltaR(q,bj))
                # associate to nearest b quark
                bj.quark = bquarks[0]
                for q in bquarks[1:]:
                    if deltaR(q,bj) < deltaR(bj.quark,bj): bj.quark = q
                ret["n_b"] += 1
                # b1 if this is matched to a quark (and thus coming from a top), b2 if from an anti-quark
                ret["i_b%d" % (1 if bj.quark.pdgId > 0 else 2)] = ib
                #print "final quark: pt %6.1f eta %+.3f phi %+.3f --> dr = %.3f" % (bj.quark.pt,bj.quark.eta,bj.quark.phi,deltaR(bj.quark,bj))
            else:
                ret["n_other_b"] += 1
        # --------------------------------------------------------------------------
        # get light jets
        for iw,ij in (1,1),(1,2),(2,1),(2,2): ret["i_w%d_j%d" % (iw,ij)] = -1
        wjtmp = []
        for ij,wj in enumerate(jets):
            # skip b-jets from true b's, done already
            if wj.mcMatchId == 6 and wj.mcMatchFlav == 5: continue
            # now set quark to none
            wj.quark = None
            # and process jets from top but not from b
            if not(wj.mcMatchId == 6 and wj.mcMatchFlav != 5): continue
            #print "jet %2d at pt %6.1f eta %+.3f phi %+.3f (%d, %d, %+2d)" % (ij,wj.pt,wj.eta,wj.phi,wj.mcMatchId,wj.mcMatchFlav,wj.mcFlavour)
            for iq,q in enumerate(lquarks):
                dr = deltaR(q,wj)
                #print "   quark %2d at pt %6.1f eta %+.3f phi %+.3f --> dr = %.3f" % (iq,q.pt,q.eta,q.phi,dr)
                if dr < 1.0 and (wj.quark == None or deltaR(wj.quark,wj) > dr):
                    wj.quark = q 
            if wj.quark:
                wjtmp.append((ij,wj))
                ret["n_w_j"] += 1
        # --------------------------------------------------------------------------
        # correctly map jets to decay chains
        wp_j = [None,None]; t1 = []
        wm_j = [None,None]; t2 = []
        for (ij,wj) in wjtmp:
            if  wj.quark.pdgId in [2,4,6]: # up or charm
                wp_j[0] = wj; ret["i_w1_j1"] = ij
            elif wj.quark.pdgId in [-1,-3,-5]: # anti-down or anti-strange (or anti-b, if any)
                wp_j[1] = wj; ret["i_w1_j2"] = ij
            elif wj.quark.pdgId in [1,3,5]: # down or strange
                wm_j[0] = wj; ret["i_w2_j1"] = ij
            elif wj.quark.pdgId in [-2,-4,-6]: # anti-up or anti-charm
                wm_j[1] = wj; ret["i_w2_j2"] = ij
            else:
                print "WTF? %s" %  [ij,wj,wj.quark.pdgId]
        def mswap(m,k1,k2):
            v1 = m[k1]; v2 = m[k2]
            m[k1] = v2; m[k2] = v1
        if wp_j[0] != None and wp_j[1] != None:
            ret["n_w_jj"] += 1
            if ret["i_w1_j1"] > ret["i_w1_j2"]: 
                mswap(wp_j, 0, 1)
                mswap(ret, "i_w1_j1", "i_w1_j2")
            if ret["i_b1"] != -1: 
                ret["n_t_jjb"] += 1
                t1 = [ bjets[ret["i_b1"]], wp_j ]
        if wm_j[0] != None and wm_j[1] != None:
            ret["n_w_jj"] += 1
            if ret["i_w2_j1"] > ret["i_w2_j2"]: 
                mswap(wm_j, 0, 1)
                mswap(ret, "i_w2_j1", "i_w2_j2")
            if ret["i_b2"] != -1: 
                ret["n_t_jjb"] += 1
                t2 = [ bjets[ret["i_b2"]], wm_j ]
        # --------------------------------------------------------------------------
        return dict([("mc_"+name,val) for (name,val) in ret.iteritems() if name in self.branches])


class TTHadCandidate:
    def __init__(self,idx,b1,b2,j11,j12,j21,j22):
        self.idx = idx
        self.b1 = b1;  
        self.b2 = b2;  
        self.j11 = j11;  
        self.j12 = j12;  
        self.j21 = j21;  
        self.j22 = j22;  
        self.w1p4 = self.j11.p4() + self.j12.p4()
        self.w2p4 = self.j21.p4() + self.j22.p4()
        self.t1p4 = self.w1p4 + self.b1.p4()
        self.t2p4 = self.w2p4 + self.b2.p4()

class TTHadEventReco:
    def __init__(self,sortersToUse={"BestGuess":""},debug=False):
        self._debug = debug
        self.sortersToUse = sortersToUse
        self.branches = [ "n_cands" ]
        self.sorters = [ ("BestGuess", BestBySingleFunc(lambda c : -abs(c.w1p4.M()-80.4)-abs(c.t1p4.M()-173)-abs(c.w2p4.M()-80.4)-abs(c.t2p4.M()-173))), ]
        self.scores = dict([(k,0) for k,s in self.sorters])
        self.scores["IDEAL"] = 0
        self.retbranches = []
        for s,postfix in self.sortersToUse.iteritems():
            self.retbranches += [ x+postfix for x in self.branches ]
    def listBranches(self):
        return self.retbranches
    def __call__(self,event):
        isMC = (event.run == 1)
        ## prepare output container
        ret  = dict([(name,0.0) for name in self.retbranches])
        # make python lists as Collection does not support indexing in slices
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        njet = len(jets); 
        if njet < 6: return ret
        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
        if len(bjets) < 2: bjets = [ jets[0], jets[1] ]
        nbjet = len(bjets); 
        # --------------------------------------------------------------------------
        info = { "n_cands": 0 }
        ttcands = []; ibingo = []
        # ---How we process one candidate-------------------------------------------
        def bingoW(ij1,ij2):
            if event.mc_i_w1_j1 == ij1 and event.mc_i_w1_j2 == ij2: return 1
            if event.mc_i_w2_j1 == ij1 and event.mc_i_w2_j2 == ij2: return 2
            return 0
        def bingoT(ib,ij1,ij2):
            if event.mc_i_b1 == ib and bingoW(ij1,ij2) == 1: return 1
            if event.mc_i_b2 == ib and bingoW(ij1,ij2) == 2: return 2 
            return 0
        def doit(ib1,ib2,ij11,ij12,ij21,ij22):
            info["n_cands"] += 1
            ttc = TTHadCandidate(info["n_cands"],bjets[ib1],bjets[ib2],jets[ij11],jets[ij12],jets[ij21],jets[ij22])
            ttcands.append(ttc)
            if isMC:
                info["good_w1"] = bingoW(ij11,ij12) 
                info["good_w2"] = bingoW(ij21,ij22) 
                info["good_t1"] = bingoT(ib1,ij11,ij12) 
                info["good_t2"] = bingoT(ib2,ij21,ij22) 
                bingo = (info["good_t1"] or info["good_t2"]) and (info["good_w1"] and info["good_w2"]) 
                if bingo: ibingo.append(info["n_cands"])
                YN={ True:"Y", False:"n", 1:'Y', 2:'Y', 0:'n' }
                if self._debug: print "candidate %3d:  b1 %d (csv %.3f, pt %6.1f) w1 %d,%d (m = %6.1f, dm = %5.1f, %1s; mt = %6.1f, dm = %5.1f, %1s)   b2 %d (csv %.3f, pt %6.1f) w2 %d,%d (m = %6.1f, dm = %5.1f, %1s; mt = %6.1f, dm = %5.1f, %1s)  %s " % ( info["n_cands"], 
                        ib1, bjets[ib1].btagCSV, bjets[ib1].pt,
                        ij11,ij12, ttc.w1p4.M(), abs(ttc.w1p4.M()-80.4), YN[info["good_w1"]],
                        ttc.t1p4.M(), abs(ttc.t1p4.M()-173), YN[info["good_t1"]],
                        ib2, bjets[ib2].btagCSV, bjets[ib2].pt,
                        ij21,ij22, ttc.w2p4.M(), abs(ttc.w2p4.M()-80.4), YN[info["good_w2"]],
                        ttc.t2p4.M(), abs(ttc.t2p4.M()-173), YN[info["good_t2"]],
                        "<<<<=== BINGO " if bingo else "")
        # --------------------------------------------------------------------------
        for ib1 in xrange(nbjet-1):
          for ib2 in xrange(ib1+1,nbjet):
            b1 = bjets[ib1]
            b2 = bjets[ib2]
            mask = [ (min(deltaR(j,b1),deltaR(j,b2)) < 0.01) for j in jets ]
            # first pair
            for ij11 in xrange(njet-1):  
              if mask[ij11]: continue
              mask[ij11] = True
              for ij12 in xrange(ij11+1, njet):  
                if mask[ij12]: continue
                mask[ij12] = True
                # second pair
                for ij21 in xrange(njet-1):  
                  if mask[ij21]: continue
                  mask[ij21] = True
                  for ij22 in xrange(ij21+1, njet):  
                    if mask[ij22]: continue
                    # process the combinaiton, with the two possible pairings of b-jets and w's
                    doit(ib1,ib2,ij11,ij12,ij21,ij22)
                    doit(ib2,ib1,ij11,ij12,ij21,ij22)
                  # done ij21
                  mask[ij21] = False
                # done ij12
                mask[ij12] = False
              # done ij11
              mask[ij11] = False
        if ibingo != []: self.scores['IDEAL'] += 1
        for sn,s in self.sorters:
            best = s(ttcands)
            if self._debug: print "Sorter %-20s selects candidate %d (%1s)" % (sn, best.idx, "Y" if best.idx in ibingo else "n")
            self.scores[sn] += (best.idx in ibingo)
        ret["n_cands"] = info["n_cands"] 
        # --------------------------------------------------------------------------
        return ret 


if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.mc = TTHadEventReco_MC()
            self.r  = TTHadEventReco(debug=True)
            self._acc = [0, 0,0, 0,0,]
        def analyze(self,ev):
            if ev.nJet25 < 6 or ev.nLepGood != 2 or abs(ev.mZ1-91.2) < 15: 
                return False
            ret = self.mc(ev)
            self._acc[0] += 1 
            self._acc[1] += (ret['mc_n_w_jj'] >= 1)
            self._acc[2] += (ret['mc_n_w_jj'] >= 2)
            self._acc[3] += (ret['mc_n_t_jjb'] >= 1)
            self._acc[4] += (ret['mc_n_t_jjb'] >= 2)
            print self._acc
            if ret["mc_n_w_jj"] < 2 or ret['mc_n_t_jjb'] < 1: return False
            for k,v in ret.iteritems(): setattr(ev,k,v)
            if self._acc[0] < 200:
                print "\nrun %6d lumi %4d event %d: leps %d, bjets %d, jets %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood, ev.nBJetMedium25, ev.nJet25)
                for k in sorted(ret.keys()):
                    print "\t%-20s: %9.3f" % (k,float(ret[k]))
            ret = self.r(ev)
            if self._acc[0] < 200:
                for k in sorted(ret.keys()):
                    print "\t%-20s: %9.3f" % (k,float(ret[k]))
    test = Tester("tester")              
    el = EventLoop([ test ])
    el.loop([tree], maxEvents = 20000)
    for k,v in test.r.scores.iteritems():
        print "%-20s: %7d" % (k,v) 
        
