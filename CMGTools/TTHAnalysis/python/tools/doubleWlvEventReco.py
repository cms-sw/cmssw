from CMGTools.TTHAnalysis.treeReAnalyzer import *
def cosThetaStar(p4l, p4nu, p4b):
    p4w = p4l + p4nu;
    p4t = p4w + p4b
    boostW = -p4w.BoostVector()
    p4lStar = ROOT.TLorentzVector(p4l);
    p4lStar.Boost(boostW)
    p4t.Boost(boostW)
    return p4lStar.Vect().Unit().Dot(p4t.Vect().Unit())

class TTEventReco_2LOS_MC:
    def __init__(self):
        self.branches = [ "met_res_x", "met_res_y", "cosThetaStar_wp", "cosThetaStar_wm", "good_tt_2los",
                          "good_tt_2los_rec", "mtop_rec_wp_gennu", "mtop_rec_wm_gennu",
                          "mlb_wp_good", "mlb_wm_good", "mlb_wp_bad", "mlb_wm_bad",
                          "cosThetaStar_wp_fit", "cosThetaStar_wm_fit",
                          "fit_deta_nu", "fit_deta_nubar", "fit_dphi_nu", "fit_dphi_nubar", "fit_dptr_nu", "fit_dptr_nubar",
                        ]
        ROOT.gSystem.Load("libFWCoreFWLite.so");
        ROOT.AutoLibraryLoader.enable()
        self.twoMTopFitter = ROOT.cmg.TwoMTopNeutrinoFitter();
    def listBranches(self):
        return [ "mc_"+x for x in self.branches ]
    def __call__(self,event):
        ## prepare output container
        ret  = dict([(name,0.0) for name in self.branches])
        ## return if not on MC
        ret0 = dict([("mc_"+name,0.0) for name in self.branches])
        if event.run != 1:   return ret0
        (met, metphi)  = event.met, event.met_phi
        metp4 = ROOT.TLorentzVector()
        metp4.SetPtEtaPhiM(met,0,metphi,0)
        bquarks = [j for j in Collection(event,"GenBQuark","nGenBQuarks",2)]
        tquarks = [j for j in Collection(event,"GenTop")]
        leps    = [j for j in Collection(event,"GenLep")]
        if len(tquarks) != 2 or len(leps) != 2: return ret0
        # pair the top (pdgId 6) with the lep+ (pdgId < 0) and the b (pdgId > 0)
        wp_chain = ( [ t for t in tquarks if t.pdgId == +6] +
                     [ l for l in leps    if l.pdgId  <  0] +
                     [ b for b in bquarks if b.pdgId == +5] ) 
        wm_chain = ( [ t for t in tquarks if t.pdgId == -6] +
                     [ l for l in leps    if l.pdgId  >  0] +
                     [ b for b in bquarks if b.pdgId == -5] ) 
        if len(wp_chain) != 3 or len(wm_chain) != 3: 
            return ret0
            raise RuntimeError, "l(wp) = %d, l(wm) = %d" % (len(wp_chain), len(wm_chain))
        wp_p4nu = wp_chain[0].p4() - wp_chain[1].p4() - wp_chain[2].p4()
        wm_p4nu = wm_chain[0].p4() - wm_chain[1].p4() - wm_chain[2].p4()
        ret["cosThetaStar_wp"] = cosThetaStar(wp_chain[1].p4(), wp_p4nu, wp_chain[2].p4()) 
        ret["cosThetaStar_wm"] = cosThetaStar(wm_chain[1].p4(), wm_p4nu, wm_chain[2].p4()) 
        ret["met_res_x"] = (wp_p4nu.Px() + wm_p4nu.Px() - metp4.Px())
        ret["met_res_y"] = (wp_p4nu.Py() + wm_p4nu.Py() - metp4.Py())
        ret["good_tt_2los"] = 1
        leps = [l for l in Collection(event,"LepGood","nLepGood",4) if l.mcMatchId == 6]
        jets = [j for j in Collection(event,"Jet","nJet25",8) if j.mcMatchId == 6]
        if len(leps) >= 2 and len(jets) >= 2:
            quarks = [ wp_chain[2], wm_chain[2] ]
            wp_chain[1].match = closest(wp_chain[1], leps)
            wm_chain[1].match = closest(wm_chain[1], leps)
            wp_chain[2].match = closest(wp_chain[2], jets)
            wm_chain[2].match = closest(wm_chain[2], jets)
            if max(wp_chain[1].match[1], wm_chain[1].match[1]) < 0.4 and max(wp_chain[2].match[1],wm_chain[2].match[1]) < 0.6:
                ret["good_tt_2los_rec"] = 1
                ret["mtop_rec_wp_gennu"] = (wp_chain[1].match[0].p4() + wp_chain[2].match[0].p4()  + wp_p4nu).M()
                ret["mtop_rec_wm_gennu"] = (wm_chain[1].match[0].p4() + wm_chain[2].match[0].p4()  + wm_p4nu).M()
                ret["mlb_wp_good"] = (wp_chain[1].match[0].p4() + wp_chain[2].match[0].p4()).M()
                ret["mlb_wm_good"] = (wm_chain[1].match[0].p4() + wm_chain[2].match[0].p4()).M()
                ret["mlb_wp_bad" ] = (wp_chain[1].match[0].p4() + wm_chain[2].match[0].p4()).M()
                ret["mlb_wm_bad" ] = (wm_chain[1].match[0].p4() + wp_chain[2].match[0].p4()).M()
                self.twoMTopFitter.initMET(event.met, event.met_phi, event.htJet25)
                self.twoMTopFitter.initLep(wp_chain[1].match[0].p4(), wm_chain[1].match[0].p4())
                self.twoMTopFitter.initBJets(wp_chain[2].match[0].p4(), wm_chain[2].match[0].p4())        
                self.twoMTopFitter.fit()
                #print "mW+ = %6.2f   mW- = %6.2f     mt = %6.2f   mtb = %6.2f " % (self.twoMTopFitter.wp().M(), self.twoMTopFitter.wm().M(), self.twoMTopFitter.t().M(), self.twoMTopFitter.tbar().M())
                ret["fit_deta_nu"   ] = wp_p4nu.Eta() - self.twoMTopFitter.nu().Eta()
                ret["fit_deta_nubar"] = wm_p4nu.Eta() - self.twoMTopFitter.nubar().Eta()
                ret["fit_dphi_nu"   ] = deltaPhi(wp_p4nu.Phi(), self.twoMTopFitter.nu().Phi())
                ret["fit_dphi_nubar"] = deltaPhi(wm_p4nu.Phi(), self.twoMTopFitter.nubar().Phi())
                ret["fit_dptr_nu"   ] = (wp_p4nu.Pt()-self.twoMTopFitter.nu().Pt())/wp_p4nu.Pt()
                ret["fit_dptr_nubar"] = (wm_p4nu.Pt()-self.twoMTopFitter.nubar().Pt())/wm_p4nu.Pt()
                ret["cosThetaStar_wp_fit"] = cosThetaStar(wp_chain[1].match[0].p4(), self.twoMTopFitter.nu(),    wp_chain[2].match[0].p4())
                ret["cosThetaStar_wm_fit"] = cosThetaStar(wm_chain[1].match[0].p4(), self.twoMTopFitter.nubar(), wm_chain[2].match[0].p4())
        return dict([("mc_"+name,val) for (name,val) in ret.iteritems()])

if __name__ == '__main0__':
    ROOT.gSystem.Load("libFWCoreFWLite.so");
    ROOT.AutoLibraryLoader.enable()
    from ROOT import TLorentzVector
    fitter = ROOT.cmg.BaseNeutrinoFitter();
    lp = TLorentzVector()
    lm = TLorentzVector()
    lm.SetPtEtaPhiM(40,0.5,1.2,0.105);
    for lpt in (40.0,   25.0, 30.0, 35.0, 50.0, 60.0):
        for leta in (0., -1.0, 1.0, -0.5, 0.5, 1.5, 2.5):
            for lphi in (0., 1., -1., 2., -2.):
                lp.SetPtEtaPhiM(lpt, leta, lphi, 0.105)
                fitter.initLep(lp,lm)
                for cth in (0., -0.4, 0.4, -0.7, -0.7):
                    for phi in (0., 1., -1., 2., -2.):
                        fitter.nll(cth,phi,0,0)
                        print "%.0f %+.1f %+.1f  %+.1f %+.1f    %.3f " % (lpt,leta,lphi,cth,phi,fitter.wp().M())
                        #exit(0)
if __name__ == '__main__':
    from sys import argv
    file = ROOT.TFile(argv[1])
    tree = file.Get("ttHLepTreeProducerBase")
    class Tester(Module):
        def __init__(self, name, booker = None):
            Module.__init__(self, name, booker)
            self.tree = PyTree(booker.book("TTree","t","")) if booker else None
            self.mc = TTEventReco_2LOS_MC()
            if self.tree:
                for V in self.mc.listBranches(): self.tree.branch(V, "F")
                self.tree.branch("htJet25", "F")
        def analyze(self,ev):
            if ev.nJet25 < 2 or ev.nLepGood > 2:
                return False
            ret = self.mc(ev)
            if not ret["mc_good_tt_2los_rec"]: return False
            if self.tree:
                self.tree.htJet25 = ev.htJet25
                for k,v in ret.iteritems(): setattr(self.tree,k,v)
            self.tree.fill()
            #print "\nrun %6d lumi %4d event %d: leps %d, bjets %d, jets %d" % (ev.run, ev.lumi, ev.evt, ev.nLepGood, ev.nBJetMedium25, ev.nJet25)
            #for k in sorted(ret.keys()):
            #    print "\t%-20s: %9.3f" % (k,float(ret[k]))
            #exit(0)
    booker = Booker("doubleWlvEventReco.root")
    test = Tester("tester", booker)              
    el = EventLoop([ test ])
    el.loop([tree], maxEvents = 20000)
    booker.done()
