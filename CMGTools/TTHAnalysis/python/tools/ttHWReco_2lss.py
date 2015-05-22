from CMGTools.TTHAnalysis.treeReAnalyzer import *

def mt(*x): 
    ht = sum([xi.Pt() for xi in x])
    pt = sum(x[1:],x[0]).Pt()
    return sqrt(max(ht*ht-pt*pt,0))

class TTHWReco_MC:
    def __init__(self):
        self.branches = [ "n_l", "n_bj", "n_qj", "has_t_jj", "has_t_jjb", "has_h_jj", 
                          "m_t_jj",  "m_h_jj", 
                          "m_t_jjb", "m_t_lb", "m_h_jjl", 
                          "pt_t_jj",  "pt_h_jj", 
                          "pt_t_jjb", "pt_t_lb", "pt_h_jjl", 
                          "mt_h_jjlv", "mt_t_lbv",
                          "i_tl_l", "i_h_l", "i_th_b", "i_tl_b", "i_t_wj1", "i_t_wj2", "i_h_wj1", "i_h_wj2",
                        ]
    def listBranches(self):
        return [ "mc_"+x for x in self.branches ]
    def __call__(self,event):
        ## prepare output container
        ret  = dict([(name,-1.0 if name[:2] == "i_" else 0.0) for name in self.branches])
        ## return if not on MC
        ret0 = dict([("mc_"+name,val) for (name,val) in ret.iteritems()])
        if event.run != 1:   return ret0
        if event.nJet25 < 3: return ret0
        # make python lists as Collection does not support indexing in slices
        leps = [l for l in Collection(event,"LepGood","nLepGood",4)]
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
        fjets = [j for j in Collection(event,"JetFwd","nJet25Fwd",3)]
        #if len(jets) < 6 and len(fjets) > 0: jets.append(fjets[0])
        if len(bjets) == 0: bjets.append(jets[0])
        (met, metphi)  = event.met, event.met_phi
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)
        njet = len(jets); nb = len(bjets); nlep = len(leps)
        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuarks",2)]
        tquarks = [j for j in Collection(event,"GenTop")]
        if len(tquarks) != 2: return ret0
        lquarks = [j for j in Collection(event,"GenQuark")]
        # --------------------------------------------------------------------------
        ret["n_l"]  = sum([l.mcMatchId > 0 for l in leps])
        ret["n_bj"] = sum([j.mcMatchId == 6 and j.mcMatchFlav == 5 for j in bjets])
        ret["n_qj"] = sum([j.mcMatchId >  0 and j.mcMatchFlav != 5 for j in jets])
        for ib, bj in enumerate(bjets):
            bj.quark = None
            if bj.mcMatchId == 6 and bj.mcMatchFlav == 5: 
                # associate to nearest b quark
                bj.quark = bquarks[0]
                for q in bquarks[1:]:
                    if deltaR(q,bj) < deltaR(bj.quark,bj): bj.quark = q
        # --------------------------------------------------------------------------
        lh, lt = None, None 
        for il,l in enumerate(leps):
            if l.mcMatchId == 6: 
                lt = l; ret["i_tl_l"] = il
            elif l.mcMatchId == 25:
                lh = l; ret["i_h_l"] = il
                continue
        for ij,wj in enumerate(jets):
            # skip b-jets from true b's, done already. Also, skip jets not from top, higgs
            if wj.mcMatchId == 6 and wj.mcMatchFlav == 5: continue
            if wj.mcMatchId == 0: continue
            # now set quark to none
            wj.quark = None
            #print "jet %2d at pt %6.1f eta %+.3f phi %+.3f (%d, %d, %+2d)" % (ij,wj.pt,wj.eta,wj.phi,wj.mcMatchId,wj.mcMatchFlav,wj.mcFlavour)
            for iq,q in enumerate(lquarks):
                dr = deltaR(q,wj)
                #print "   quark %2d at pt %6.1f eta %+.3f phi %+.3f --> dr = %.3f" % (iq,q.pt,q.eta,q.phi,dr)
                if dr < 1.0 and (wj.quark == None or deltaR(wj.quark,wj) > dr):
                    wj.quark = q 
            if wj.quark != None:
                if wj.mcMatchId == 6:
                    if ret["i_t_wj1"] != -1:
                        ret["i_t_wj2"] = ij
                    else:
                        ret["i_t_wj1"] = ij
                elif wj.mcMatchId == 25:
                    if ret["i_h_wj1"] != -1:
                        ret["i_h_wj2"] = ij
                    else:
                        ret["i_h_wj1"] = ij
        ret["has_t_jj"] = (ret["i_t_wj2"] != -1)
        ret["has_h_jj"] = (ret["i_h_wj2"] != -1)
        tqWlv,tqWjjb = None,None
        for t in tquarks:
            t.bquark = None
            ## attach the b quark to it
            for b in bquarks:
                if b.pdgId * t.pdgId > 0: t.bquark = b ## they're both quarks or both anti-quarks
            ## test if Wlv or Wjjb
            if lt != None and t.pdgId * lt.charge > 0: ## the top (id>0) gives a positively charged lepton
                tqWlv = t
            else: 
                tqWjjb = t
        for ib,b in enumerate(bjets):
            if b.quark == None: continue 
            if tqWlv != None and b.quark == tqWlv.bquark:
                ret["i_tl_b"] = ib
            if tqWjjb != None and b.quark == tqWjjb.bquark:
                ret["i_th_b"] = ib
        ret["has_t_jjb"]  = (ret["i_th_b"] != -1) and ret["has_t_jj"]
        p4_t_jj = jets[ret["i_t_wj1"]].p4() + jets[ret["i_t_wj2"]].p4() if ret["has_t_jj"] else None
        p4_h_jj = jets[ret["i_h_wj1"]].p4() + jets[ret["i_h_wj2"]].p4() if ret["has_h_jj"] else None
        p4_t_jjb = p4_t_jj + bjets[ret["i_th_b"]].p4() if ret["has_t_jjb"] else None
        p4_h_jjl = p4_h_jj + leps[ret["i_h_l"]].p4() if (ret["has_h_jj"] and ret["i_h_l"] != -1) else None
        mt_h_jjlv = mt(metp4, jets[ret["i_h_wj1"]].p4(), jets[ret["i_h_wj2"]].p4(), leps[ret["i_h_l"]].p4())  if (ret["has_h_jj"] and ret["i_h_l"] != -1) else None
        mt_t_lvb  = mt(metp4, bjets[ret["i_tl_b"]].p4(),  leps[ret["i_tl_l"]].p4())  if (ret["i_tl_b"] != -1 and ret["i_tl_l"] != -1) else None 
        ret["m_t_jj"] = p4_t_jj.M() if p4_t_jj else -99.0
        ret["m_h_jj"] = p4_h_jj.M() if p4_h_jj else -99.0
        ret["m_t_jjb"] = p4_t_jjb.M() if p4_t_jjb else -99.0
        ret["m_h_jjl"] = p4_h_jjl.M() if p4_h_jjl else -99.0
        ret["pt_t_jj"] = p4_t_jj.Pt() if p4_t_jj else -99.0
        ret["pt_h_jj"] = p4_h_jj.Pt() if p4_h_jj else -99.0
        ret["pt_t_jjb"] = p4_t_jjb.Pt() if p4_t_jjb else -99.0
        ret["pt_h_jjl"] = p4_h_jjl.Pt() if p4_h_jjl else -99.0
        ret["mt_h_jjlv"] = mt_h_jjlv if mt_h_jjlv else -99.0
        ret["mt_h_jjlv"] = mt_h_jjlv if mt_h_jjlv else -99.0
        ret["mt_t_lvb"]  = mt_t_lvb if mt_t_lvb else -99.0
        # --------------------------------------------------------------------------
        return dict([("mc_"+name,val) for (name,val) in ret.iteritems() if name in self.branches])


##
##class TTCandidate:
##    def __init__(self,idx,bj,lW,lb,j1,j2,tjjb,metp4):
##        self.idx = idx
##        self.bj = bj; self.lW = lW; self.lb = lb; self.j1 = j1; self.j2 = j2; self.tjjb = tjjb; 
##        p4_bl = bj.p4() if tjjb == False else lb.p4()*(1.0/lb.jetPtRatio)
##        p4_bh = bj.p4() if tjjb == True  else lb.p4()*(1.0/lb.jetPtRatio)
##        pz1, pz2 = solveWlv(lW,metp4.Pt(),metp4.Phi())
##        pzmin = pz1 if abs(pz1) < abs(pz2) else pz2
##        self.metp4min = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pzmin, hypot(metp4.Pt(),pzmin))
##        self.p4wlv  = self.metp4min + lW.p4()
##        self.p4wjj  = j1.p4() + j2.p4()
##        self.p4tjjb = self.p4wjj + p4_bh
##        self.mtWlv  = mt(lW.p4(),metp4)
##        self.p4tlvb = self.metp4min + lW.p4() + p4_bl
##        self.mwjjErr = self.p4wjj.M() * hypot(jetEtResolution(j1)/max(j1.pt,25),  jetEtResolution(j2)/max(j2.pt,25))
##        self.mttlvb  = mt(metp4, lW.p4(), p4_bl)
##def bestByX(cands,score):
##        #print "called bestByX with N(c) = %d " % len(cands)
##        selected  = [cands[0]]; 
##        bestscore = score(cands[0])
##        for c in cands[1:]:
##            s = score(c)
##            if s > bestscore:
##                selected  = [c]
##                bestscore = s 
##            elif s == bestscore:
##                selected += [c]
##        #print "%s -> %s (score %.4f)" % ( [ s.idx for s in cands ], [ s.idx for s in selected ], bestscore)
##        #print " --- returning N(c) = %d " % len(selected)
##        return selected
##
##class BestByWtWjjBTagMlvb:
##    def __call__(self,cands):
##        csorted = cands[:]
##        csorted = bestByX(csorted, lambda c : -abs(c.mtWlv-80.4))
##        csorted = bestByX(csorted, lambda c : -abs(c.p4wjj.M()-80.4))
##        csorted = bestByX(csorted, lambda c : c.bj.btagCSV)
##        csorted = bestByX(csorted, lambda c : -abs(c.p4tlvb.M()-172.5))
##        return csorted[0] 
##class BestBySingleFunc:
##    def __init__(self,func):
##        self.func = func
##    def __call__(self,cands):
##        csorted = bestByX(cands, self.func)
##        return csorted[0] 
##class BestByCascade:
##    def __init__(self,*funcs):
##        self.funcs = funcs
##    def __call__(self,cands):
##        csorted = cands[:]
##        for func in self.funcs:
##            #print "%d --> " % len(csorted),
##            csorted = bestByX(csorted, func)
##            #print "%d. "% len(csorted)
##        return csorted[0] 
##
##
##class TTEventReco:
##    def __init__(self,sortersToUse={"BestGuess":""},debug=False):
##        self._debug = debug
##        self.sortersToUse = sortersToUse
##        self.branches = [ "mt_Wlv", "good_Wlv",
##                          "v_pz1", "v_pz2",
##                          "W_pz1", "W_pz2",
##                          "t_pz1", "t_pz2",
##                          "has_Wjj",   "m_Wjj", "pt_Wjj", "good_Wjj", 
##                          "has_tjjb",  "has_tjjlb", "m_tjjb", "pt_tjjb",  
##                          "has_tlvb",  "has_tlvlb", "m_tlvb", "mt_tlvb", "pt_tlvb", "m_tlvb1", "m_tlvb2",
##                          "RecTT2LSSW1_pt", "RecTT2LSSW1_eta", "RecTT2LSSW1_phi", "RecTT2LSSW1_mass", "RecTT2LSSW1_pdgId",
##                          "RecTT2LSSW2_pt", "RecTT2LSSW2_eta", "RecTT2LSSW2_phi", "RecTT2LSSW2_mass", "RecTT2LSSW2_pdgId",
##                          "RecTT2LSS1_pt", "RecTT2LSS1_eta", "RecTT2LSS1_phi", "RecTT2LSS1_mass", "RecTT2LSS1_pdgId",
##                          "RecTT2LSS2_pt", "RecTT2LSS2_eta", "RecTT2LSS2_phi", "RecTT2LSS2_mass", "RecTT2LSS2_pdgId",
##                          "n_cands",
##                        ]
##        self.sorters = [
##                         #("BestByWt",BestByCascade((lambda c : -abs(c.mtWlv-80.4)))),
##                         #("BestByWlt",BestByCascade((lambda c : -TTLikelihood_MC('mc_mt_Wlv',c.mtWlv)))),
##                         #("BestByWtBTag",BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : c.bj.btagCSV))),
##                         #("BestByWtBTag",BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : c.bj.btagCSV))),
##                         #("BestByWtBPt", BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : c.bj.pt))),
##                         #("BestByWtMlvb",BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : -999*c.tjjb-abs(c.p4tlvb.M()-172.5)))),
##                         #("BestByMjj",    BestByCascade((lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjNoB", BestByCascade((lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjNoBB",BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjNobbNoB", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.679), (lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), (lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjNoBBNoBT",BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), (lambda c : -abs(c.p4wjj.M()-80.4)))),
##                         #("BestByMjjSigNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898),
##                         #                                       (lambda c : -abs(c.p4wjj.M()-80.4)/c.mwjjErr))),
##                         #("BestByMjjLNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898),
##                         #                                       (lambda c : -TTLikelihood_MC('mc_m_Wjj',c.p4wjj.M())))),
##                         ("BestGuess", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
##                                                      (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
##                                                      (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
##                                                      (lambda c : -abs(c.p4wjj.M()-80.4)), # Best W->jj by M
##                                                      (lambda c : c.bj.pt), # Best b-jet by pt
##                                                      )),
##                         #("ByGuessLL0", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
##                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
##                         #                              (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
##                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
##                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
##                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
##                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
##                         #                                ), 
##                         #                              )),
##                         #("ByGuessLL", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
##                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
##                         #                              (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
##                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
##                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
##                         #                                          -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
##                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
##                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
##                         #                                ), 
##                         #                              )),
##                         #("ByGuessLL2", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
##                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
##                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
##                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
##                         #                                          -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
##                         #                                          -TTLikelihood_MC('mc_mt_Wlv', c.mtWlv)
##                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
##                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
##                         #                                ), 
##                         #                              )),
##                        ("ByGuessLL2B", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
##                                                      (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
##                                                      (lambda c : c.bj.pt), # Best b-jet by pt
##                                                      (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
##                                                                  -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
##                                                                  -( TTLikelihood_MC('mc_mt_Wlv', c.mtWlv) - TTLikelihood_MC('mc_mt_Wlv_wrong', c.mtWlv) )
##                                                                  -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
##                                                                  -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
##                                                        ), 
##                                                      )),
##                         #("BestByWtWjjBTagMlvb",BestByWtWjjBTagMlvb()),
##                         #("BestByTopTop",BestBySingleFunc(lambda c : -abs(c.p4tlvb.M()-172.5)-abs(c.p4tjjb.M()-172.5))),
##                         #("BestBySum4",BestBySingleFunc(lambda c : 
##                         #   -abs((c.p4wjj.M()-80.4)/15) -abs((c.mtWlv-80.4)/20)
##                         #   -abs((c.p4tlvb.M()-172.5)/30) -abs((c.p4tjjb.M()-172.5)/30) ) ),
##                         #("BestBySum4NoTJJb",BestBySingleFunc(lambda c : 
##                         #   -999*c.tjjb
##                         #   -abs((c.p4wjj.M()-80.4)/15) -abs((c.mtWlv-80.4)/20)
##                         #   -abs((c.p4tlvb.M()-172.5)/30) -abs((c.p4tjjb.M()-172.5)/30) ) ),
##                       ]
##        self.scores = dict([(k,0) for k,s in self.sorters])
##        self.scores["IDEAL"] = 0
##        self.retbranches = []
##        for s,postfix in self.sortersToUse.iteritems():
##            self.retbranches += [ x+postfix for x in self.branches ]
##    def listBranches(self):
##        return self.retbranches
##    def __call__(self,event):
##        isMC = (event.run == 1)
##        ## prepare output container
##        ret  = dict([(name,0.0) for name in self.retbranches])
##        # make python lists as Collection does not support indexing in slices
##        leps = [l for l in Collection(event,"LepGood","nLepGood",4)]
##        jets = [j for j in Collection(event,"Jet","nJet25",8)]
##        njet = len(jets); nlep = len(leps)
##        if njet < 3 or nlep < 2: return ret
##        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
##        if len(bjets) == 0: bjets.append(jets[0])
##        nb = len(bjets)
##        (met, metphi)  = event.met, event.met_phi
##        metp4 = ROOT.TLorentzVector()
##        metp4.SetPtEtaPhiM(met,0,metphi,0)
##        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuarks",2)] if isMC else []
##        tquarks = [j for j in Collection(event,"GenTop")]   if isMC else []
##        lquarks = [j for j in Collection(event,"GenQuark")] if isMC else []
##        # --------------------------------------------------------------------------
##        info = { "n_cands": 0 }
##        ttcands = []; ibingo = []
##        for ib,bj in enumerate(bjets):
##            for ilW,lW in enumerate(leps):
##                for ilb,lb in enumerate(leps):
##                    if ilW == ilb: continue
##                    for ij1,j1 in enumerate(jets):
##                        if deltaR(j1,bj) < 0.01: continue 
##                        for ij2,j2 in enumerate(jets):
##                            if ij2 <= ij1 or deltaR(j2,bj) < 0.01: continue
##                            for tjjb in False, True:
##                                info["n_cands"] += 1
##                                ttc = TTCandidate(info["n_cands"],bj,lW,lb,j1,j2,tjjb,metp4)
##                                ttcands.append(ttc)
##                                if isMC:
##                                    info["good_b"  ] = (ib  == event.mc_i_b)
##                                    info["good_lb" ] = (ilb == event.mc_i_lb)
##                                    info["good_Wlv"] = (ilW == event.mc_i_lW)
##                                    info["good_Wjj"] = (ij1 == min(event.mc_i_wj1,event.mc_i_wj2) and ij2 == max(event.mc_i_wj1,event.mc_i_wj2))
##                                    info["good_tjjb"]  = event.mc_has_tjjb  and info["good_b" ] and info["good_Wjj"] and tjjb
##                                    info["good_tjjlb"] = event.mc_has_tjjlb and info["good_lb"] and info["good_Wjj"] and not tjjb
##                                    info["good_tlvb"]  = event.mc_has_tlvb  and info["good_b" ] and info["good_Wlv"] and not tjjb
##                                    info["good_tlvlb"] = event.mc_has_tlvlb and info["good_lb"] and info["good_Wlv"] and tjjb
##                                    bingo = info["good_b"] and info["good_Wlv"] and info["good_Wjj"] and (info["good_tjjb"] or info["good_tjjlb"]) and (info["good_tlvb"] or info["good_tlvlb"])
##                                    #bingo = info["good_Wjj"]
##                                    if bingo: ibingo.append(info["n_cands"])
##                                    YN={ True:"Y", False:"n" }
##                                    if self._debug: print "candidate %3d:  ib %d (%.3f, %1s) ilW %d (%1s) tjjb %1s   m(Wjj%d%d) =%6.1f (%5.1f, %1s)    mT(Wlv) =%6.1f (%5.1f, %1s)   m(tjjb) =%6.1f (%5.1f, %1s)   m(tlvb) =%6.1f (%5.1f, %1s) %s " % ( info["n_cands"],
##                                            ib, bj.btagCSV, YN[info["good_b"]],
##                                            ilW, YN[info["good_Wlv"]],
##                                            YN[tjjb], 
##                                            ij1,ij2,ttc.p4wjj.M(), abs(ttc.p4wjj.M()-80.4), YN[info["good_Wjj"]],
##                                            ttc.mtWlv, abs(ttc.mtWlv-80.4), YN[info["good_Wlv"]],
##                                            ttc.p4tjjb.M(), abs(ttc.p4tjjb.M()-172.5), YN[info["good_tjjb"] or info["good_tjjlb"]],
##                                            ttc.p4tlvb.M(), abs(ttc.p4tlvb.M()-172.5), YN[info["good_tlvb"] or info["good_tlvlb"]],
##                                            "<<<<=== BINGO " if bingo else "")
##        if ibingo != []: self.scores['IDEAL'] += 1
##        for sn,s in self.sorters:
##            best = s(ttcands)
##            if self._debug: print "Sorter %-20s selects candidate %d (%1s)" % (sn, best.idx, YN[best.idx in ibingo])
##            self.scores[sn] += (best.idx in ibingo)
##            if sn not in self.sortersToUse: continue
##            postfix = self.sortersToUse[sn]
##            ret["mt_Wlv"+postfix]  = best.mtWlv
##            ret["m_Wjj"+postfix]   = best.p4wjj.M()
##            ret["pt_Wjj"+postfix]  = best.p4wjj.Pt()
##            ret["m_tlvb"+postfix]  = best.p4tlvb.M()
##            ret["mt_tlvb"+postfix] = best.mttlvb
##            ret["pt_tlvb"+postfix] = best.p4tlvb.Pt()
##            ret["m_tjjb"+postfix]  = best.p4tjjb.M()
##            #ret["mt_tjjb"+postfix] = best.p4tjjb.Pt()
##            ret["pt_tjjb"+postfix]  = best.p4tjjb.Pt()
##            if postfix == "":
##                ret["RecTT2LSS1_pt"  ] = best.p4tlvb.Pt()
##                ret["RecTT2LSS1_eta" ] = best.p4tlvb.Eta()
##                ret["RecTT2LSS1_phi" ] = best.p4tlvb.Phi()
##                ret["RecTT2LSS1_mass"] = best.p4tlvb.M()
##                ret["RecTT2LSS1_pdgId" ] = +6 if best.lW.charge > 0 else -6
##                ret["RecTT2LSS2_pt"  ] = best.p4tjjb.Pt()
##                ret["RecTT2LSS2_eta" ] = best.p4tjjb.Eta()
##                ret["RecTT2LSS2_phi" ] = best.p4tjjb.Phi()
##                ret["RecTT2LSS2_mass"] = best.p4tjjb.M()
##                ret["RecTT2LSS2_pdgId" ] = -6 if best.lW.charge > 0 else +6
##                ret["RecTT2LSSW1_pt"  ] = best.p4wlv.Pt()
##                ret["RecTT2LSSW1_eta" ] = best.p4wlv.Eta()
##                ret["RecTT2LSSW1_phi" ] = best.p4wlv.Phi()
##                ret["RecTT2LSSW1_mass"] = best.p4wlv.M()
##                ret["RecTT2LSSW1_pdgId" ] = +24 if best.lW.charge > 0 else -24
##                ret["RecTT2LSSW2_pt"  ] = best.p4wjj.Pt()
##                ret["RecTT2LSSW2_eta" ] = best.p4wjj.Eta()
##                ret["RecTT2LSSW2_phi" ] = best.p4wjj.Phi()
##                ret["RecTT2LSSW2_mass"] = best.p4wjj.M()
##                ret["RecTT2LSSW2_pdgId" ] = -24 if best.lW.charge > 0 else +24
##
##        ret["n_cands"] = info["n_cands"] 
##        # --------------------------------------------------------------------------
##        return ret 
##

if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.mc = TTHWReco_MC()
            self.counts = [0]*6
        def analyze(self,ev):
            i = 0
            if ev.nJet25 <= 3 or ev.nBJetMedium25 < 1 or ev.nLepGood > 2 or abs(ev.LepGood1_pdgId+ev.LepGood2_pdgId) < 26 or ev.GenHiggsDecayMode != 24:
                return False
            self.counts[i] += 1; i += 1 
            if ev.nJet25 <= 5 or ev.nBJetMedium25 < 2:
                return False
            ret = self.mc(ev)
            self.counts[i] += 1; i += 1 
            if ret["mc_n_l"] < 2: return False
            if ret["mc_n_bj"] < 1: return False
            self.counts[i] += 1; i += 1 
            if ret["mc_has_t_jjb"] == 0: return False
            self.counts[i] += 1; i += 1 
            if ret["mc_has_h_jj"] == 0: return False
            self.counts[i] += 1; i += 1 
            for k,v in ret.iteritems(): setattr(ev,k,v)
            print "\nrun %6d lumi %4d event %d: leps %d, bjets %d, jets %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood, ev.nBJetMedium25, ev.nJet25)
            for k in sorted(ret.keys()):
                print "\t%-20s: %9.3f" % (k,float(ret[k]))
            print self.counts
    test = Tester("tester")              
    el = EventLoop([ test ])
    el.loop([tree], maxEvents = 20000)
