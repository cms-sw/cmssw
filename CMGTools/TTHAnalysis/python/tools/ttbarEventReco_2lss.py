from CMGTools.TTHAnalysis.treeReAnalyzer import *

def mt(*x): 
    ht = sum([xi.Pt() for xi in x])
    pt = sum(x[1:],x[0]).Pt()
    return sqrt(max(ht*ht-pt*pt,0))
def solveWlv(lW,met,metphi):
    MW=80.4
    a = (1 - (lW.p4().Z()/lW.p4().E())**2)
    ppe    = met * lW.pt * cos(lW.phi - metphi)/lW.p4().E()
    brk    = MW**2 / (2*lW.p4().E()) + ppe
    b      = (lW.p4().Z()/lW.p4().E()) * brk
    c      = met**2 - brk**2
    delta   = b**2 - a*c
    sqdelta = sqrt(delta)    if delta    > 0 else 0
    return [ (b + s*sqdelta)/a for s in +1,-1 ]

class TTEventReco_MC:
    def __init__(self):
        self.branches = [ "has_b", "has_2b", "is_b_1", "has_lfake", "has_lb",
                          "n_extraj", "n_wj_b", "has_wj_l", "has_extraj_l",
                          "b_ptr", "b_dr", "lb_ptr", "lb_ptr_raw", "lb_dr",
                          "has_Wlv",   "mt_Wlv", "mt_Wlv_wrong", "m_Wlv_genv", "mt_Wlv_genv",
                          "v_pz1", "v_pz2", "v_pz_gen",
                          "W_pz1", "W_pz2", "W_pz_gen",
                          "t_pz1", "t_pz2", "t_pz_gen",
                          "has_Wjj_gen", "has_Wjj",   "m_Wjj", "pt_Wjj", "m_Wjj_gen", 
                          "has_tjjb",  "has_tjjlb", "m_tjjb", "pt_tjjb", "m_tjjb_gen", "m_tjjb_genw", "m_tjjb_genb", 
                          "has_tlvb",  "has_tlvlb", "m_tlvb", "mt_tlvb", "pt_tlvb", "m_tlvb1", "m_tlvb2", "m_tlvb_good_v_pz",
                          "RecTT2LSSW1_pt", "RecTT2LSSW1_eta", "RecTT2LSSW1_phi", "RecTT2LSSW1_mass", "RecTT2LSSW1_pdgId", 
                          "RecTT2LSSW2_pt", "RecTT2LSSW2_eta", "RecTT2LSSW2_phi", "RecTT2LSSW2_mass", "RecTT2LSSW2_pdgId", 
                          "RecTT2LSS1_pt", "RecTT2LSS1_eta", "RecTT2LSS1_phi", "RecTT2LSS1_mass", "RecTT2LSS1_pdgId", 
                          "RecTT2LSS2_pt", "RecTT2LSS2_eta", "RecTT2LSS2_phi", "RecTT2LSS2_mass", "RecTT2LSS2_pdgId", 
                          "i_b", "i_wj1", "i_wj2", "i_lW", "i_lb",
                        ]
    def listBranches(self):
        return [ "mc_"+x for x in self.branches ]
    def __call__(self,event):
        ## prepare output container
        ret  = dict([(name,0.0) for name in self.branches])
        ## return if not on MC
        ret0 = dict([("mc_"+name,0.0) for name in self.branches])
        if event.run != 1:   return ret0
        if event.nJet25 < 3: return ret0
        # make python lists as Collection does not support indexing in slices
        leps = [l for l in Collection(event,"LepGood","nLepGood",4)]
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
        if len(bjets) == 0: bjets.append(jets[0])
        (met, metphi)  = event.met_pt, event.met_phi
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)
        njet = len(jets); nb = len(bjets); nlep = len(leps)
        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuark")]
        tquarks = [j for j in Collection(event,"GenTop")]
        if len(tquarks) != 2: return ret0
        lquarks = [j for j in Collection(event,"GenQuark")]
        # --------------------------------------------------------------------------
        ret["has_b"]  = (sum([j.mcMatchId == 6 and j.mcMatchFlav == 5 for j in bjets]) > 0)
        ret["has_2b"]  = (sum([j.mcMatchId == 6 and j.mcMatchFlav == 5 for j in bjets]) > 1)
        ret["is_b_1"] = (nb >= 1 and bjets[0].mcMatchId == 6 and bjets[0].mcMatchFlav == 5)
        ret["n_extraj"] = len([j for j in jets if j.mcMatchId != 6 and j.mcMatchId != 25])
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
                ret["b_dr"] = deltaR(bj.quark,bj)
                ret["b_ptr"] = bj.pt/bj.quark.pt
                ret["i_b"] = ib
                #print "final quark: pt %6.1f eta %+.3f phi %+.3f --> dr = %.3f" % (bj.quark.pt,bj.quark.eta,bj.quark.phi,deltaR(bj.quark,bj))
            elif bj.mcMatchId == 6 and bj.mcMatchFlav != 5:
                ret["n_wj_b"] += 1
        # --------------------------------------------------------------------------
        lW, lb = None, None 
        for il,l in enumerate(leps):
            if l.mcMatchId == 6: 
                lW = l
                ret["i_lW"] = il
                continue
            if l.mcMatchId == 0 and l.mcMatchAny == 2:
                for q in bquarks:
                    if deltaR(l,q) < 0.8:
                        if lb != None and deltaR(lb.quark,lb) < deltaR(l,q): 
                            continue
                        lb = l
                        lb.quark = q
                        ret["lb_ptr_raw"] = lb.pt/lb.quark.pt
                        ret["lb_ptr"] = (lb.pt/lb.jetPtRatio)/lb.quark.pt
                        ret["lb_dr"]  = deltaR(lb,lb.quark)
                        ret["i_lb"] = il
        if lb == None: # do we have a w->j->l or it's really fake
           for l in leps:
              if l.mcMatchId > 0: continue
              matched = False
              for j in lquarks:
                 if deltaR(l,j) < 0.5: 
                    matched = True
                    ret["has_wj_l"] += 1
              if not matched: ret["has_extraj_l"] += 1
        ret["has_lfake"] = (sum([l.mcMatchId == 0 for l in leps])> 0)
        #ret["has_lb"] =  (sum([l.mcMatchId == 0 and l.mcMatchAny == 2 for l in leps])> 0)
        ret["has_lb"] = (lb != None)
        # --------------------------------------------------------------------------
        ret["has_Wlv"] = (lW != None)
        ret["mt_Wlv"]  = sqrt(2*lW.pt*met*(1-cos(lW.phi-metphi))) if lW else 0.0
        ret["mt_Wlv_wrong"]  = sqrt(2*lb.pt*met*(1-cos(lb.phi-metphi))) if lb else 0.0
        # --------------------------------------------------------------------------
        #ret["has_same_b"] = (lb != None) and (sum([b.quark == lb.quark for b in bjets]) > 0)
        # --------------------------------------------------------------------------
        wjets = [j for j in jets if j.mcMatchId == 6 and j.mcMatchFlav != 5]
        for iq in 0,1: ret["i_wj%d" % (iq+1)] = -1
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
                    ret["i_wj%d" % (iq+1)] = ij
        ret["has_Wjj"] = (len(wjets) == 2)
        qjacc =  [q for q in lquarks if q.pt > 25 and abs(q.eta) < 2.4]
        ret["has_Wjj_gen"] = (len(qjacc) == 2)
        ret["has_drjj_gen"] = (deltaR(qjacc[0],qjacc[1]) if len(qjacc) == 2 else -1.0)
        if len(wjets) == 2:
            wjjp4 =  wjets[0].p4()+wjets[1].p4()
            ret["m_Wjj"] = wjjp4.M()
            ret["pt_Wjj"] = wjjp4.Pt()
            if wjets[0].quark and wjets[1].quark:
                ret["m_Wjj_gen"] = (wjets[0].quark.p4() + wjets[1].quark.p4()).M()
        # --------------------------------------------------------------------------
        tqWlv, tqWjjb = None, None
        for t in tquarks:
            t.bquark = None
            ## attach the b quark to it
            for b in bquarks:
                if b.pdgId * t.pdgId > 0: t.bquark = b ## they're both quarks or both anti-quarks
            ## test if Wlv or Wjjb
            if lW != None and t.pdgId * lW.charge > 0: ## the top (id>0) gives a positively charged lepton
                tqWlv = t
            else: 
                tqWjjb = t
        #ret["has_gen_tqWlv"]  = (tqWlv  != None)
        #ret["has_gen_tqWjjb"] = (tqWjjb != None)
        # --------------------------------------------------------------------------
        if tqWjjb != None and ret["has_b"] and ret["has_Wjj"]:
            t = tqWjjb
            b_p4 = None
            for bj in bjets:
                if t.bquark != None and bj.quark == t.bquark:
                    ret["has_tjjb"] = True
                    b_p4 = bj.p4()
                    tqWjjb.rec_b   = lb
            if lb != None:
                if lb.quark == t.bquark:
                    ret["has_tjjlb"] = True
                    b_p4 = lb.p4()*(1.0/lb.jetPtRatio)
                    tqWjjb.rec_b = bj
            if b_p4 != None:                
                tjjbp4 = wjets[0].p4() + wjets[1].p4() + b_p4 
                ret["m_tjjb"]  = tjjbp4.M()
                ret["pt_tjjb"] = tjjbp4.Pt()
                ret["m_tjjb_genb"] = (wjets[0].p4() + wjets[1].p4() + t.bquark.p4()).M()
                if wjets[0].quark and wjets[1].quark:
                    ret["m_tjjb_gen"]  = (wjets[0].quark.p4() + wjets[1].quark.p4() + tqWjjb.bquark.p4()).M()
                    ret["m_tjjb_genw"] = (wjets[0].quark.p4() + wjets[1].quark.p4() + b_p4).M()
                #tqWjjb.rec_Wjj = wjets
                #tqWjjb.rec_p4 = tjjbp4
                ret["RecTT2LSS2_pt"  ] = tjjbp4.Pt()
                ret["RecTT2LSS2_eta" ] = tjjbp4.Eta()
                ret["RecTT2LSS2_phi" ] = tjjbp4.Phi()
                ret["RecTT2LSS2_mass"] = tjjbp4.M()
                ret["RecTT2LSS2_pdgId" ] = tqWjjb.pdgId
                ret["RecTT2LSSW2_pt"  ]   = (wjets[0].p4() + wjets[1].p4()).Pt()
                ret["RecTT2LSSW2_eta" ]   = (wjets[0].p4() + wjets[1].p4()).Eta()
                ret["RecTT2LSSW2_phi" ]   = (wjets[0].p4() + wjets[1].p4()).Phi()
                ret["RecTT2LSSW2_mass"]   = (wjets[0].p4() + wjets[1].p4()).M()
                ret["RecTT2LSSW2_pdgId" ] = 0 # -24 if  > 0 else +24

        # --------------------------------------------------------------------------
        MW=80.1
        if tqWlv != None and ret["has_Wlv"]:
            t = tqWlv
            ret["tlv_bq_eta"] = t.bquark.eta if t.bquark != None else 99.0
            ret["tlv_bq_phi"] = t.bquark.phi if t.bquark != None else 99.0
            ## solve quadratic equation for W mass
            pz1, pz2 = solveWlv(lW,met,metphi)
            ret["v_pz1"] = pz1; ret["v_pz2"] = pz2
            ret["W_pz1"] = lW.p4().Z()+pz1; ret["W_pz2"] = lW.p4().Z()+pz2
            metp4z1 = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pz1, hypot(met,pz1))
            metp4z2 = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pz2, hypot(met,pz2))
            ## take the minimum, for now
            pzmin = pz1 if abs(pz1) < abs(pz2) else pz2
            metp4min = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pzmin, hypot(met,pzmin))
            ## now the rest 
            b_p4 = None; v_p4 = None
            for bj in bjets:
                if bj.mcMatchId == 6 and bj.mcMatchFlav == 5:
                    ret["bj_bq_eta"] = bj.quark.eta if bj.quark != None else 99.0
                    ret["bj_bq_phi"] = bj.quark.phi if bj.quark != None else 99.0
                if t.bquark != None and bj.quark == t.bquark:
                    ret["has_tlvb"] = True
                    v_p4 = t.p4() - bj.quark.p4() - lW.p4()
                    b_p4 = bj.p4()
            if lb != None:
                ret["lb_bq_eta"] = lb.quark.eta if lb.quark != None else 99.0
                ret["lb_bq_phi"] = lb.quark.phi if lb.quark != None else 99.0
                if lb.quark == t.bquark:
                    ret["has_tlvlb"] = True
                    v_p4 = t.p4() - lb.quark.p4() - lW.p4()
                    b_p4 = lb.p4()*(1.0/lb.jetPtRatio)
            if b_p4 != None:
                ret["m_Wlv_genv"] = (v_p4 + lW.p4()).M()
                ret["mt_Wlv_genv"] = mt(v_p4, lW.p4())
                ret["v_pz_gen"] = v_p4.Z()
                ret["W_pz_gen"] = v_p4.Z() + lW.p4().Z()
                ret["t_pz_gen"] = v_p4.Z() + lW.p4().Z() + b_p4.Z()
                ret["t_pz1"] = lW.p4().Z() + pz1 + b_p4.Z()
                ret["t_pz2"] = lW.p4().Z() + pz2 + b_p4.Z()
                pzcheat = pz1 if abs(pz1-v_p4.Z()) < abs(pz2-v_p4.Z()) else pz2
                metp4cheat = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pzcheat, hypot(met,pzcheat))
                ret["m_tlvb1"]          = (metp4z1    + lW.p4() + b_p4).M()
                ret["m_tlvb2"]          = (metp4z2    + lW.p4() + b_p4).M()
                ret["m_tlvb"]           = (metp4min   + lW.p4() + b_p4).M()
                ret["m_tlvb_good_v_pz"] = (metp4cheat + lW.p4() + b_p4).M()
                ret["pt_tlvb"] = (metp4 + lW.p4() + b_p4).Pt()
                ret["mt_tlvb"] = mt(metp4, lW.p4(), b_p4)
                ret["RecTT2LSS1_pt"  ] = (metp4min + lW.p4() + b_p4).Pt()
                ret["RecTT2LSS1_eta" ] = (metp4min + lW.p4() + b_p4).Eta()
                ret["RecTT2LSS1_phi" ] = (metp4min + lW.p4() + b_p4).Phi()
                ret["RecTT2LSS1_mass"] = (metp4min + lW.p4() + b_p4).M()
                ret["RecTT2LSS1_pdgId" ] = t.pdgId
                ret["RecTT2LSSW1_pt"  ]   = (metp4min + lW.p4()).Pt()
                ret["RecTT2LSSW1_eta" ]   = (metp4min + lW.p4()).Eta()
                ret["RecTT2LSSW1_phi" ]   = (metp4min + lW.p4()).Phi()
                ret["RecTT2LSSW1_mass"]   = (metp4min + lW.p4()).M()
                ret["RecTT2LSSW1_pdgId" ] = 0 # -24 if  > 0 else +24

        # --------------------------------------------------------------------------
        return dict([("mc_"+name,val) for (name,val) in ret.iteritems() if name in self.branches])


def jetEtResolution(jet):
   """ pfjet resolutions. taken from AN-2010-371
   "   http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/abenagli/Fall10/NtuplePackage/src/TKinFitterUtils.cc?revision=1.2&view=markup """
   Et = max(jet.pt,25)
   Eta = jet.eta
   N, S, C, m = (0,0,0,0);
   if abs(Eta) < 0.5:
       N = 3.96859;
       S = 0.18348;
       C = 0.;
       m = 0.62627;
   elif  abs(Eta) < 1.:
       N = 3.55226;
       S = 0.24026;
       C = 0.;
       m = 0.52571;
   elif  abs(Eta) < 1.5:
       N = 4.54826;
       S = 0.22652;
       C = 0.;
       m = 0.58963;
   elif  abs(Eta) < 2.:
       N = 4.62622;
       S = 0.23664;
       C = 0.;
       m = 0.48738;
   elif  abs(Eta) < 2.5:
       N = 2.53324;
       S = 0.34306;
       C = 0.;
       m = 0.28662;
   return (N * abs(N) ) + (S * S) * pow(Et, m+1) + (C * C) * Et * Et;


class TTLikelihood:
    def __init__(self,filename):
        self._file = ROOT.TFile(filename)
    def __call__(self,histname,val,proc="TT",eps=0.0002):
        hist = self._file.Get(histname+"_"+proc)
        if not bool(hist): raise RuntimeError, "Could not find %s " % (histname+"_"+proc)
        norm = hist.Integral()
        val  = hist.GetBinContent( min(max(hist.GetXaxis().FindBin(val), 1),hist.GetNbinsX()) )
        return - 2 * log ( max(val/norm, eps) )
#TTLikelihood_MC = TTLikelihood("/afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/python/plotter/plots/250513/tmp/topReco/2mu/2lss_topReco_plots.root")
TTLikelihood_MC = TTLikelihood("/afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/python/plotter/plots/250513/tests/topreco/2lss/mumu/fullgen/2lss_topReco_plots.root")


class TTCandidate:
    def __init__(self,idx,bj,lW,lb,j1,j2,tjjb,metp4):
        self.idx = idx
        self.bj = bj; self.lW = lW; self.lb = lb; self.j1 = j1; self.j2 = j2; self.tjjb = tjjb; 
        p4_bl = bj.p4() if tjjb == False else lb.p4()*(1.0/lb.jetPtRatio)
        p4_bh = bj.p4() if tjjb == True  else lb.p4()*(1.0/lb.jetPtRatio)
        pz1, pz2 = solveWlv(lW,metp4.Pt(),metp4.Phi())
        pzmin = pz1 if abs(pz1) < abs(pz2) else pz2
        self.metp4min = ROOT.TLorentzVector(metp4.Px(), metp4.Py(), pzmin, hypot(metp4.Pt(),pzmin))
        self.p4wlv  = self.metp4min + lW.p4()
        self.p4wjj  = j1.p4() + j2.p4()
        self.p4tjjb = self.p4wjj + p4_bh
        self.mtWlv  = mt(lW.p4(),metp4)
        self.p4tlvb = self.metp4min + lW.p4() + p4_bl
        self.mwjjErr = self.p4wjj.M() * hypot(jetEtResolution(j1)/max(j1.pt,25),  jetEtResolution(j2)/max(j2.pt,25))
        self.mttlvb  = mt(metp4, lW.p4(), p4_bl)
def bestByX(cands,score):
        #print "called bestByX with N(c) = %d " % len(cands)
        selected  = [cands[0]]; 
        bestscore = score(cands[0])
        for c in cands[1:]:
            s = score(c)
            if s > bestscore:
                selected  = [c]
                bestscore = s 
            elif s == bestscore:
                selected += [c]
        #print "%s -> %s (score %.4f)" % ( [ s.idx for s in cands ], [ s.idx for s in selected ], bestscore)
        #print " --- returning N(c) = %d " % len(selected)
        return selected

class BestByWtWjjBTagMlvb:
    def __call__(self,cands):
        csorted = cands[:]
        csorted = bestByX(csorted, lambda c : -abs(c.mtWlv-80.4))
        csorted = bestByX(csorted, lambda c : -abs(c.p4wjj.M()-80.4))
        csorted = bestByX(csorted, lambda c : c.bj.btagCSV)
        csorted = bestByX(csorted, lambda c : -abs(c.p4tlvb.M()-172.5))
        return csorted[0] 
class BestBySingleFunc:
    def __init__(self,func):
        self.func = func
    def __call__(self,cands):
        csorted = bestByX(cands, self.func)
        return csorted[0] 
class BestByCascade:
    def __init__(self,*funcs):
        self.funcs = funcs
    def __call__(self,cands):
        csorted = cands[:]
        for func in self.funcs:
            #print "%d --> " % len(csorted),
            csorted = bestByX(csorted, func)
            #print "%d. "% len(csorted)
        return csorted[0] 


class TTEventReco:
    def __init__(self,sortersToUse={"BestGuess":""},debug=False):
        self._debug = debug
        self.sortersToUse = sortersToUse
        self.branches = [ "mt_Wlv", "good_Wlv",
                          "v_pz1", "v_pz2",
                          "W_pz1", "W_pz2",
                          "t_pz1", "t_pz2",
                          "has_Wjj",   "m_Wjj", "pt_Wjj", "good_Wjj", 
                          "has_tjjb",  "has_tjjlb", "m_tjjb", "pt_tjjb",  
                          "has_tlvb",  "has_tlvlb", "m_tlvb", "mt_tlvb", "pt_tlvb", "m_tlvb1", "m_tlvb2",
                          "RecTT2LSSW1_pt", "RecTT2LSSW1_eta", "RecTT2LSSW1_phi", "RecTT2LSSW1_mass", "RecTT2LSSW1_pdgId",
                          "RecTT2LSSW2_pt", "RecTT2LSSW2_eta", "RecTT2LSSW2_phi", "RecTT2LSSW2_mass", "RecTT2LSSW2_pdgId",
                          "RecTT2LSS1_pt", "RecTT2LSS1_eta", "RecTT2LSS1_phi", "RecTT2LSS1_mass", "RecTT2LSS1_pdgId",
                          "RecTT2LSS2_pt", "RecTT2LSS2_eta", "RecTT2LSS2_phi", "RecTT2LSS2_mass", "RecTT2LSS2_pdgId",
                          "n_cands",
                        ]
        self.sorters = [
                         #("BestByWt",BestByCascade((lambda c : -abs(c.mtWlv-80.4)))),
                         #("BestByWlt",BestByCascade((lambda c : -TTLikelihood_MC('mc_mt_Wlv',c.mtWlv)))),
                         ("BestByWtBTag",BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : c.bj.btagCSV))),
                         ("BestByWtBPt", BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : c.bj.pt))),
                         #("BestByWtMlvb",BestByCascade((lambda c : -abs(c.mtWlv-80.4)),(lambda c : -999*c.tjjb-abs(c.p4tlvb.M()-172.5)))),
                         #("BestByMjj",    BestByCascade((lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjNoB", BestByCascade((lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjNoBB",BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjNobbNoB", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.679), (lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), (lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjNoBBNoBT",BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.676), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), (lambda c : -abs(c.p4wjj.M()-80.4)))),
                         #("BestByMjjSigNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898),
                         #                                       (lambda c : -abs(c.p4wjj.M()-80.4)/c.mwjjErr))),
                         #("BestByMjjLNobbNoBT", BestByCascade((lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898),
                         #                                       (lambda c : -TTLikelihood_MC('mc_m_Wjj',c.p4wjj.M())))),
                         ("BestGuess", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
                                                      (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
                                                      (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
                                                      (lambda c : -abs(c.p4wjj.M()-80.4)), # Best W->jj by M
                                                      (lambda c : c.bj.pt), # Best b-jet by pt
                                                      )),
                         #("ByGuessLL0", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
                         #                              (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
                         #                                ), 
                         #                              )),
                         #("ByGuessLL", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
                         #                              (lambda c : -abs(c.mtWlv-80.4)), # Best W->lv by MT
                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
                         #                                          -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
                         #                                ), 
                         #                              )),
                         #("ByGuessLL2", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
                         #                              (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
                         #                              (lambda c : c.bj.pt), # Best b-jet by pt
                         #                              (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
                         #                                          -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
                         #                                          -TTLikelihood_MC('mc_mt_Wlv', c.mtWlv)
                         #                                          -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
                         #                                          -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
                         #                                ), 
                         #                              )),
                        ("ByGuessLL2B", BestByCascade((lambda c : -999*c.tjjb), ## no tjjb
                                                      (lambda c : min(c.j1.btagCSV,c.j2.btagCSV) < 0.244), (lambda c : max(c.j1.btagCSV,c.j2.btagCSV) < 0.898), # No W->bb, W->Bx
                                                      (lambda c : c.bj.pt), # Best b-jet by pt
                                                      (lambda c : -TTLikelihood_MC('mc_pt_Wjj',c.p4wjj.Pt()) 
                                                                  -TTLikelihood_MC('mc_m_Wjj', c.p4wjj.M())
                                                                  -( TTLikelihood_MC('mc_mt_Wlv', c.mtWlv) - TTLikelihood_MC('mc_mt_Wlv_wrong', c.mtWlv) )
                                                                  -TTLikelihood_MC('mc_toptop_ptb', (c.p4tlvb.Pt()-c.p4tjjb.Pt())/(c.p4tlvb.Pt()-c.p4tjjb.Pt())) 
                                                                  -TTLikelihood_MC('mc_toptop_dphi',abs(deltaPhi(c.p4tjjb.Phi(), c.p4tlvb.Phi())))
                                                        ), 
                                                      )),
                         #("BestByWtWjjBTagMlvb",BestByWtWjjBTagMlvb()),
                         #("BestByTopTop",BestBySingleFunc(lambda c : -abs(c.p4tlvb.M()-172.5)-abs(c.p4tjjb.M()-172.5))),
                         ("BestBySum4",BestBySingleFunc(lambda c : 
                            -abs((c.p4wjj.M()-80.4)/15) -abs((c.mtWlv-80.4)/20)
                            -abs((c.p4tlvb.M()-172.5)/30) -abs((c.p4tjjb.M()-172.5)/30) ) ),
                         #("BestBySum4NoTJJb",BestBySingleFunc(lambda c : 
                         #   -999*c.tjjb
                         #   -abs((c.p4wjj.M()-80.4)/15) -abs((c.mtWlv-80.4)/20)
                         #   -abs((c.p4tlvb.M()-172.5)/30) -abs((c.p4tjjb.M()-172.5)/30) ) ),
                       ]
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
        leps = [l for l in Collection(event,"LepGood","nLepGood",4)]
        jets = [j for j in Collection(event,"Jet","nJet25",8)]
        njet = len(jets); nlep = len(leps)
        if njet < 3 or nlep < 2: return ret
        bjets = [ j for j in jets if j.btagCSV > 0.679 ]
        if len(bjets) == 0: 
            jsorted = jets[:]
            jsorted.sort(key=lambda j:j.btagCSV) 
            bjets.append(jsorted[-1])
        nb = len(bjets)
        (met, metphi)  = event.met_pt, event.met_phi
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)
        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuark")] if isMC else []
        tquarks = [j for j in Collection(event,"GenTop")]   if isMC else []
        lquarks = [j for j in Collection(event,"GenQuark")] if isMC else []
        # --------------------------------------------------------------------------
        info = { "n_cands": 0 }
        ttcands = []; ibingo = []
        for ib,bj in enumerate(bjets):
            for ilW,lW in enumerate(leps):
                for ilb,lb in enumerate(leps):
                    if ilW == ilb: continue
                    for ij1,j1 in enumerate(jets):
                        if deltaR(j1,bj) < 0.01: continue 
                        for ij2,j2 in enumerate(jets):
                            if ij2 <= ij1 or deltaR(j2,bj) < 0.01: continue
                            for tjjb in False, True:
                                info["n_cands"] += 1
                                ttc = TTCandidate(info["n_cands"],bj,lW,lb,j1,j2,tjjb,metp4)
                                ttcands.append(ttc)
                                if isMC:
                                    info["good_b"  ] = (ib  == event.mc_i_b)
                                    info["good_lb" ] = (ilb == event.mc_i_lb)
                                    info["good_Wlv"] = (ilW == event.mc_i_lW)
                                    info["good_Wjj"] = (ij1 == min(event.mc_i_wj1,event.mc_i_wj2) and ij2 == max(event.mc_i_wj1,event.mc_i_wj2))
                                    info["good_tjjb"]  = event.mc_has_tjjb  and info["good_b" ] and info["good_Wjj"] and tjjb
                                    info["good_tjjlb"] = event.mc_has_tjjlb and info["good_lb"] and info["good_Wjj"] and not tjjb
                                    info["good_tlvb"]  = event.mc_has_tlvb  and info["good_b" ] and info["good_Wlv"] and not tjjb
                                    info["good_tlvlb"] = event.mc_has_tlvlb and info["good_lb"] and info["good_Wlv"] and tjjb
                                    bingo = info["good_b"] and info["good_Wlv"] and info["good_Wjj"] and (info["good_tjjb"] or info["good_tjjlb"]) and (info["good_tlvb"] or info["good_tlvlb"])
                                    #bingo = info["good_Wlv"] and (info["good_tlvb"] or info["good_tlvlb"])
                                    #bingo = info["good_Wjj"]  and (info["good_tjjb"] or info["good_tjjlb"])
                                    if bingo: ibingo.append(info["n_cands"])
                                    YN={ True:"Y", False:"n" }
                                    if self._debug: print "candidate %3d:  ib %d (%.3f, %1s) ilW %d (%1s) tjjb %1s   m(Wjj%d%d) =%6.1f (%5.1f, %1s)    mT(Wlv) =%6.1f (%5.1f, %1s)   m(tjjb) =%6.1f (%5.1f, %1s)   m(tlvb) =%6.1f (%5.1f, %1s) %s " % ( info["n_cands"],
                                            ib, bj.btagCSV, YN[info["good_b"]],
                                            ilW, YN[info["good_Wlv"]],
                                            YN[tjjb], 
                                            ij1,ij2,ttc.p4wjj.M(), abs(ttc.p4wjj.M()-80.4), YN[info["good_Wjj"]],
                                            ttc.mtWlv, abs(ttc.mtWlv-80.4), YN[info["good_Wlv"]],
                                            ttc.p4tjjb.M(), abs(ttc.p4tjjb.M()-172.5), YN[info["good_tjjb"] or info["good_tjjlb"]],
                                            ttc.p4tlvb.M(), abs(ttc.p4tlvb.M()-172.5), YN[info["good_tlvb"] or info["good_tlvlb"]],
                                            "<<<<=== BINGO " if bingo else "")
        if ibingo != []: self.scores['IDEAL'] += 1
        for sn,s in self.sorters:
            best = s(ttcands)
            if self._debug: print "Sorter %-20s selects candidate %d (%1s)" % (sn, best.idx, YN[best.idx in ibingo])
            self.scores[sn] += (best.idx in ibingo)
            if sn not in self.sortersToUse: continue
            postfix = self.sortersToUse[sn]
            ret["mt_Wlv"+postfix]  = best.mtWlv
            ret["m_Wjj"+postfix]   = best.p4wjj.M()
            ret["pt_Wjj"+postfix]  = best.p4wjj.Pt()
            ret["m_tlvb"+postfix]  = best.p4tlvb.M()
            ret["mt_tlvb"+postfix] = best.mttlvb
            ret["pt_tlvb"+postfix] = best.p4tlvb.Pt()
            ret["m_tjjb"+postfix]  = best.p4tjjb.M()
            #ret["mt_tjjb"+postfix] = best.p4tjjb.Pt()
            ret["pt_tjjb"+postfix]  = best.p4tjjb.Pt()
            if postfix == "":
                ret["RecTT2LSS1_pt"  ] = best.p4tlvb.Pt()
                ret["RecTT2LSS1_eta" ] = best.p4tlvb.Eta()
                ret["RecTT2LSS1_phi" ] = best.p4tlvb.Phi()
                ret["RecTT2LSS1_mass"] = best.p4tlvb.M()
                ret["RecTT2LSS1_pdgId" ] = +6 if best.lW.charge > 0 else -6
                ret["RecTT2LSS2_pt"  ] = best.p4tjjb.Pt()
                ret["RecTT2LSS2_eta" ] = best.p4tjjb.Eta()
                ret["RecTT2LSS2_phi" ] = best.p4tjjb.Phi()
                ret["RecTT2LSS2_mass"] = best.p4tjjb.M()
                ret["RecTT2LSS2_pdgId" ] = -6 if best.lW.charge > 0 else +6
                ret["RecTT2LSSW1_pt"  ] = best.p4wlv.Pt()
                ret["RecTT2LSSW1_eta" ] = best.p4wlv.Eta()
                ret["RecTT2LSSW1_phi" ] = best.p4wlv.Phi()
                ret["RecTT2LSSW1_mass"] = best.p4wlv.M()
                ret["RecTT2LSSW1_pdgId" ] = +24 if best.lW.charge > 0 else -24
                ret["RecTT2LSSW2_pt"  ] = best.p4wjj.Pt()
                ret["RecTT2LSSW2_eta" ] = best.p4wjj.Eta()
                ret["RecTT2LSSW2_phi" ] = best.p4wjj.Phi()
                ret["RecTT2LSSW2_mass"] = best.p4wjj.M()
                ret["RecTT2LSSW2_pdgId" ] = -24 if best.lW.charge > 0 else +24

        ret["n_cands"] = info["n_cands"] 
        # --------------------------------------------------------------------------
        return ret 





if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("treeProducerSusyMultilepton")
    tree.vectorTree = True
    class Tester(Module):
        def __init__(self, name):
            Module.__init__(self,name,None)
            self.mc = TTEventReco_MC()
            self.r  = TTEventReco(debug=True)
            self._acc = [0,0,0,0]
        def analyze(self,ev):
            if ev.nJet25 not in [3,4] or ev.nBJetMedium25 < 1 or ev.nLepGood10 != 2 or ev.LepGood_charge[0]*ev.LepGood_charge[1] < 0:
            #if ev.nJet25 <= 3 or ev.nBJetMedium25 != 1 or ev.nLepGood > 2 or abs(ev.LepGood1_pdgId+ev.LepGood2_pdgId) < 26:
                return False
            ret = self.mc(ev)
            self._acc[0] += 1 
            self._acc[1] += ret['mc_has_Wjj_gen']
            #self._acc[2] += ret['mc_has_Wjj_gen'] and ret['mc_has_drjj_gen'] > 0.5
            self._acc[3] += ret['mc_has_Wjj']
            #self._acc[1] += ret['mc_has_Wlv']
            #self._acc[2] += ret['mc_has_b']
            #self._acc[2] += ret['mc_has_Wjj_gen'] and ret['mc_has_drjj_gen'] > 0.5
            #self._acc[3] += ret['mc_has_Wjj']
            print self._acc
            #return False
            #if not ret["mc_has_Wjj"]: return False
            if not ret["mc_has_Wlv"]: return False
            #if ret["mc_mt_Wlv"] < 40: return False
            #if not ret["mc_has_tjjlb"]: return False
            #if not ret["mc_has_tlvb"]: return False
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
    el.loop([tree], maxEvents = 100000)
    for k,v in test.r.scores.iteritems():
        print "%-20s: %7d" % (k,v) 
        
