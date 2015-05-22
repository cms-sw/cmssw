import ROOT, os

from CMGTools.TTHAnalysis.plotter.rocCurves import hist2ROC1d
if "/rhoCorrections_cc.so" not in ROOT.gSystem.GetLibraries():
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/macros/leptons/rhoCorrections.cc+" % os.environ['CMSSW_BASE']);

def _histToEff(hist2d,graph,fout=None):
    cpcalc = ROOT.TEfficiency.ClopperPearson
    for b in xrange(1,hist2d.GetNbinsX()+1):
        x0 = hist2d.GetXaxis().GetBinCenter(b)
        xmin, xmax = hist2d.GetXaxis().GetBinLowEdge(b), hist2d.GetXaxis().GetBinUpEdge(b)
        passing = int(hist2d.GetBinContent(b,2))
        total   = int(hist2d.GetBinContent(b,1) + passing)
        y0 = passing/float(total) if total else -99
        ymax = cpcalc(total,passing,0.6827,True ) if total else -99
        ymin = cpcalc(total,passing,0.6827,False) if total else -99
        graph.SetPoint(b-1, x0, y0)
        graph.SetPointError(b-1, x0-xmin, xmax-x0, y0-ymin,ymax-y0)
        if fout:
            fout.write("%8.4f %8.4f     %.4f  -%.4f/+%.4f\n" % (xmin,xmax, y0, y0-ymin,ymax-y0) )

class EAFitter:
    def __init__(self,tree,options):
        self.options = options
        self.tree = tree
        self.c1 = ROOT.TCanvas("c1", "c1", 600, 600)
        self.c1.Draw()
        self.fout = ROOT.TFile.Open("%s/EA.root" % self.options.printDir, "RECREATE")
    def fitEAEtaBins(self,name,radius,selection,etabins,rhovar,rhobins,func="pol1",param=1,verbose=False):
        if ROOT.gROOT.FindObject("htemp") != None: ROOT.gROOT.FindObject("htemp").Delete()
        from array import array
        hist2d = ROOT.TProfile2D("htemp","htemp",len(etabins)-1,array('d',etabins),len(rhobins)-1,array('d',rhobins))
        hist1d = ROOT.TH1D("h1d","h1d;%s;<Iso%s>" % (rhovar,radius),len(rhobins)-1,array('d',rhobins))
        self.tree.Draw("(LepGood_absRawNeutralIso%s):(%s):abs(LepGood_eta)>>htemp" % (radius,rhovar), selection, "prof goff")
        ret = []
        for i in xrange(1,len(etabins)):
            for b in xrange(1,len(rhobins)):
                hist1d.SetBinContent(b, hist2d.GetBinContent(i,b))
                hist1d.SetBinError(b, hist2d.GetBinError(i,b))
            hist1d.Fit(func,"Q");
            if verbose:
                hist1d.GetYaxis().SetRangeUser(0,1.5*hist2d.GetMaximum())
                hist1d.GetYaxis().SetDecimals(True)
                self.c1.Print("%s/%s_eta%d.png" % (self.options.printDir, name, i) );
            myf = hist1d.GetFunction(func)
            ret.append((etabins[i-1],etabins[i], myf.GetParameter(param), myf.GetParError(param)))
        histEA = ROOT.TH1D(name,name+";|#eta|;EA",len(etabins)-1,array('d',etabins))
        fout = open("%s/%s.txt"%(self.options.printDir,name),"w")
        fout.write(" e_min  e_max    EA  +- err\n");    
        print      " e_min  e_max    EA  +- err";    
        for i,(etamin, etamax, ea,eaerr) in enumerate(ret):
            histEA.SetBinContent(i+1,ea)
            histEA.SetBinError(i+1,eaerr)
            print      " %.3f  %.3f   %6.4f  %.4f"   % (etamin,etamax,ea,eaerr)
            fout.write(" %.3f  %.3f   %6.4f  %.4f\n" % (etamin,etamax,ea,eaerr))
        histEA.GetYaxis().SetRangeUser(0,1.5*histEA.GetMaximum())
        histEA.GetYaxis().SetDecimals(True)
        histEA.Draw("E");
        self.c1.Print("%s/%s.png" % (self.options.printDir, name) );
        self.c1.Print("%s/%s.pdf" % (self.options.printDir, name) );
        self.c1.Print("%s/%s.eps" % (self.options.printDir, name) );
        self.fout.WriteTObject(histEA.Clone());
        return ret
    def doEAResiduals(self,name,radius,selection,etabins,rhovar,vtxbins,func="pol1",param=1,verbose=False):
        if ROOT.gROOT.FindObject("htemp") != None: ROOT.gROOT.FindObject("htemp").Delete()
        from array import array
        hist2d = ROOT.TProfile2D("htemp","htemp",len(etabins)-1,array('d',etabins),len(vtxbins)-1,array('d',vtxbins))
        hist1d = ROOT.TH1D("h1d","h1d;N(vtx);<max(Iso%s-#rho*EA,0)>" % (radius),len(vtxbins)-1,array('d',vtxbins))
        histEA = self.fout.Get(name)
        histR  = ROOT.TH1D(name+"R", name+";|#eta|;Residual",len(etabins)-1,array('d',etabins))
        histRR = ROOT.TH1D(name+"RR",name+";|#eta|;Residual/EA",len(etabins)-1,array('d',etabins))
        ret = []
        for i in xrange(1,len(etabins)):
            mysel = "(%s) && %f <= abs(LepGood_eta) && abs(LepGood_eta)<%f " % (selection,etabins[i-1],etabins[i])
            ea = histEA.GetBinContent(i)
            self.tree.Draw("max(LepGood_absRawNeutralIso{R}-{rho}*{ea},0):nVert:abs(LepGood_eta)>>+htemp".format(R=radius,rho=rhovar,ea=ea), mysel, "prof goff")
            for b in xrange(1,len(vtxbins)):
                hist1d.SetBinContent(b, hist2d.GetBinContent(i,b))
                hist1d.SetBinError(b, hist2d.GetBinError(i,b))
            hist1d.Fit(func,"Q");
            if verbose:
                hist1d.GetYaxis().SetRangeUser(0,1.5*hist2d.GetMaximum())
                hist1d.GetYaxis().SetDecimals(True)
                self.c1.Print("%s/%s_eta%d_residual.png" % (self.options.printDir, name, i) );
            resV = hist1d.GetFunction(func).GetParameter(param)
            resE = hist1d.GetFunction(func).GetParError(param)
            histR.SetBinContent(i, resV)
            histR.SetBinError(i, resE)
            histRR.SetBinContent(i, resV/ea)
            histRR.SetBinError(i, resE/ea)
        histR.GetYaxis().SetRangeUser(0,1.5*histR.GetMaximum())
        histRR.GetYaxis().SetRangeUser(0,1.5*histRR.GetMaximum())
        histR.GetYaxis().SetDecimals(True)
        histRR.GetYaxis().SetDecimals(True)
        histR.Draw("E");
        self.c1.Print("%s/%s_residual.png" % (self.options.printDir, name) );
        self.c1.Print("%s/%s_residual.pdf" % (self.options.printDir, name) );
        self.c1.Print("%s/%s_residual.eps" % (self.options.printDir, name) );
        histRR.Draw("E");
        self.c1.Print("%s/%s_residualrel.png" % (self.options.printDir, name) );
        self.c1.Print("%s/%s_residualrel.pdf" % (self.options.printDir, name) );
        self.c1.Print("%s/%s_residualrel.eps" % (self.options.printDir, name) );

    def done(self):
        self.fout.Close()

class EATester:
    def __init__(self,tree,options):
        self.options = options
        self.tree = tree
        self.c1 = ROOT.TCanvas("c1", "c1", 600, 600)
        self.c1.Draw()
        #self.fin = ROOT.TFile.Open("%s/EA.root" % self.options.printDir)
    def plotEffEtaEABins(self,name,radius,rhovar,eaname,cut,selection,etabins,vtxbins,yrange):
        from array import array
        ROOT.loadEAHisto("EA_el","%s/EA.root" % self.options.printDir,eaname+"_el")
        ROOT.loadEAHisto("EA_mu","%s/EA.root" % self.options.printDir,eaname+"_mu")
        hist2d = ROOT.TH2D("htemp","htemp",len(vtxbins)-1,array('d',vtxbins),2,array('d',[-0.5,0.5,1.5]))
        hist1d = ROOT.TH1D("frame","frame;N(vtx);Efficiency",len(vtxbins)-1,array('d',vtxbins))
        gr1dc  = ROOT.TGraphAsymmErrors(len(vtxbins)-1);
        gr1du  = ROOT.TGraphAsymmErrors(len(vtxbins)-1);
        gr1dd  = ROOT.TGraphAsymmErrors(len(vtxbins)-1);
        for i in xrange(1,len(etabins)):
            etamin, etamax = etabins[i-1], etabins[i]
            myname = "%s_eta_%.3f_%.3f" % (name, etamin, etamax) 
            mysel  = "(%s) && %f <= abs(LepGood_eta) &&  abs(LepGood_eta) < %f" % (selection,etamin,etamax)
            isoc="(LepGood_chargedHadRelIso{R}+eaCorr(LepGood_absRawNeutralIso{R},LepGood_pdgId,LepGood_eta,{rho})/LepGood_pt<{cut})".format(R=radius,rho=rhovar,cut=cut)
            isou="(LepGood_chargedHadRelIso{R}+LepGood_absRawNeutralIso{R}/LepGood_pt<{cut})".format(R=radius,cut=cut)
            isod="(LepGood_chargedHadRelIso{R}+max(LepGood_absRawNeutralIso{R}-0.5*LepGood_puIso{R},0)/LepGood_pt<{cut})".format(R=radius,cut=cut)
            self.tree.Draw("%s:nVert>>htemp" % isoc, mysel, "goff")
            foutc = open("%s/%s.txt"%(self.options.printDir,myname),"w")
            _histToEff(hist2d,gr1dc,foutc)
            self.tree.Draw("%s:nVert>>htemp" % isou, mysel, "goff")
            foutu = open("%s/%s_uncorr.txt"%(self.options.printDir,myname),"w")
            _histToEff(hist2d,gr1du,foutu)
            self.tree.Draw("%s:nVert>>htemp" % isod, mysel, "goff")
            foutd = open("%s/%s_dbeta.txt"%(self.options.printDir,myname),"w")
            _histToEff(hist2d,gr1dd,foutd)
            gr1dd.SetLineColor(ROOT.kRed-4);
            gr1dd.SetMarkerColor(ROOT.kRed-4);
            gr1du.SetLineColor(ROOT.kGray+1);
            gr1du.SetMarkerColor(ROOT.kGray+1);
            gr1dc.SetLineColor(ROOT.kBlue+1);
            gr1dc.SetMarkerColor(ROOT.kBlue+1);
            for j,X in enumerate([gr1dd,gr1du,gr1dc]):
                print "Fitting %s [%d]" % (myname,j)
                X.Fit("pol1","F EX0","")
                X.GetFunction("pol1").SetLineColor(X.GetLineColor())
                X.GetFunction("pol1").SetLineStyle(2)
                X.GetFunction("pol1").SetLineWidth(2)
            ROOT.gStyle.SetOptFit(False)
            hist1d.Draw("AXIS"); hist1d.GetYaxis().SetRangeUser(yrange[0],yrange[1])
            gr1du.Draw("P SAME"); 
            gr1dd.Draw("P SAME"); 
            gr1dc.Draw("P SAME");
            self.c1.Print("%s/%s.png" % (self.options.printDir, myname) );
    def plotROCEtaEABins(self,name,radius,rhovar,eaname,selsig,selbkg,etabins):
        ROOT.loadEAHisto("EA_el","%s/EA.root" % self.options.printDir,eaname+"_el")
        ROOT.loadEAHisto("EA_mu","%s/EA.root" % self.options.printDir,eaname+"_mu")
        hsig1d = ROOT.TH1D("hsig","hsig",2000,0,10)
        hbkg1d = ROOT.TH1D("hbkg","hbkg",2000,0,10)
        hist1d = ROOT.TH1D("frame","frame;Eff(background);Eff(signal)",100,0.,0.319)
        hist1d.GetYaxis().SetRangeUser(0.4,1.019)
        for i in xrange(1,len(etabins)):
            etamin, etamax = etabins[i-1], etabins[i]
            myname = "%s_eta_%.3f_%.3f" % (name, etamin, etamax) 
            mysig  = "(%s) && %f <= abs(LepGood_eta) &&  abs(LepGood_eta) < %f" % (selsig,etamin,etamax)
            mybkg  = "(%s) && %f <= abs(LepGood_eta) &&  abs(LepGood_eta) < %f" % (selbkg,etamin,etamax)
            isoc="(LepGood_chargedHadRelIso{R}+eaCorr(LepGood_absRawNeutralIso{R},LepGood_pdgId,LepGood_eta,{rho})/LepGood_pt)".format(R=radius,rho=rhovar)
            isou="(LepGood_chargedHadRelIso{R}+LepGood_absRawNeutralIso{R}/LepGood_pt)".format(R=radius)
            isod="(LepGood_chargedHadRelIso{R}+max(LepGood_absRawNeutralIso{R}-0.5*LepGood_puIso{R},0)/LepGood_pt)".format(R=radius)
            self.tree.Draw("%s>>hsig" % isoc, mysig, "goff")
            self.tree.Draw("%s>>hbkg" % isoc, mybkg, "goff")
            rocc = hist2ROC1d(hsig1d,hbkg1d)
            self.tree.Draw("%s>>hsig" % isou, mysig, "goff")
            self.tree.Draw("%s>>hbkg" % isou, mybkg, "goff")
            rocu = hist2ROC1d(hsig1d,hbkg1d)
            self.tree.Draw("%s>>hsig" % isod, mysig, "goff")
            self.tree.Draw("%s>>hbkg" % isod, mybkg, "goff")
            rocd = hist2ROC1d(hsig1d,hbkg1d)
            hist1d.Draw()
            rocd.SetLineWidth(3)
            rocu.SetLineWidth(3)
            rocc.SetLineWidth(3)
            rocd.SetLineColor(ROOT.kRed-4);
            rocu.SetLineColor(ROOT.kGray+1);
            rocc.SetLineColor(ROOT.kBlue+1);
            rocu.Draw("L SAME"); 
            rocd.Draw("L SAME"); 
            rocc.Draw("L SAME");
            self.c1.Print("%s/%s.png" % (self.options.printDir, myname) );


    

def addEAFitterOptions(parser):
    parser.add_option("-t", "--tree", dest="tree", default='treeProducerSusyMultilepton', help="Pattern for tree name");
    parser.add_option("-c", "--cut",  dest="cut", type="string", default="1", help="Lepton selection");
    parser.add_option("--pdir", "--print-dir", dest="printDir", type="string", default="plots", help="print out plots in this directory");


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] tree reftree")
    addEAFitterOptions(parser)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print "You must specify one tree to fit"
        exit()
    file = ROOT.TFile.Open(args[0])
    tree = file.Get(options.tree) 
    ROOT.gROOT.SetBatch(True)
    ROOT.gROOT.ProcessLine(".x ../../python/plotter/tdrstyle.cc")
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetErrorX(0.5)
    vtxbins = [5,10,13,15,16,17,18,19,20,22,25,30]
    if True:
        fitter = EAFitter(tree,options)
        rhobins   = [5,6,7,8,9,10,11,12,13,14,15]
        rhoCNbins = [6,7,8,9,10,11,12,13,14,15,16,17,18]
        for lname,lid in ("mu",13), ("el",11):
            sel = "(%s) && abs(LepGood_pdgId) == %d && LepGood_mcMatchId > 0 && LepGood_pt > 25 " % (options.cut, lid)
            #fitter.fitEAEtaBins("EAv1_"+lname, "03", sel, [0,0.8,1.3,2.0,2.2,2.5], "rho",rhobins,verbose=True)
            #fitter.fitEAEtaBins("EAv1CN_"+lname, "03",sel, [0,0.8,1.3,2.0,2.2,2.5], "rhoCentralNeutral", rhoCNbins, verbose=True)
            #fitter.fitEAEtaBins("fine_"+lname, "03", sel, [0.2*i for i in xrange(0,13)], "rho",rhobins, verbose=False)
            #fitter.fitEAEtaBins("fineCN_"+lname, "03",sel, [0.2*i for i in xrange(0,13)], "rhoCentralNeutral", rhoCNbins, verbose=False)
            #fitter.doEAResiduals("testCN_"+lname, "03",sel, [0,0.8,1.3,2.0,2.2,2.5], "rhoCentralNeutral", [0]+vtxbins, verbose=True)
            fitter.fitEAEtaBins("EAv1R04_"+lname,   "04", sel, [0,0.8,1.3,2.0,2.2,2.5], "rho",rhobins,verbose=True)
            fitter.fitEAEtaBins("EAv1R04CN_"+lname, "04", sel, [0,0.8,1.3,2.0,2.2,2.5], "rhoCentralNeutral", rhoCNbins, verbose=True)
            fitter.fitEAEtaBins("fineR04_"+lname,   "04", sel, [0.2*i for i in xrange(0,13)], "rho",rhobins, verbose=False)
            fitter.fitEAEtaBins("fineR04CN_"+lname, "04", sel, [0.2*i for i in xrange(0,13)], "rhoCentralNeutral", rhoCNbins, verbose=False)
        fitter.done()
    if False:
        tester = EATester(tree,options)
        for lname,lid in ("mu",13), ("el",11):
            selsig = "(%s) && abs(LepGood_pdgId) == %d && LepGood_mcMatchId >  0" % (options.cut, lid)
            seleff = "(%s) && abs(LepGood_pdgId) == %d && LepGood_mcMatchId >  0  && LepGood_pt > 25" % (options.cut, lid)
            selbkg = "(%s) && abs(LepGood_pdgId) == %d && LepGood_mcMatchId <= 0" % (options.cut, lid)
            #tester.plotEffEtaEABins( "effR03Loose_"  +lname,"03","rho",              "EAv1",  0.5,seleff,[0,1.2,2.4],vtxbins,[0.94,1.009])
            #tester.plotEffEtaEABins("fakeR03Loose_"  +lname,"03","rho",              "EAv1",  0.5,selbkg,[0,1.2,2.4],vtxbins,[0.0,1.019])
            #tester.plotEffEtaEABins( "effR03LooseCN_"+lname,"03","rhoCentralNeutral","EAv1CN",0.5,seleff,[0,1.2,2.4],vtxbins,[0.94,1.009])
            #tester.plotEffEtaEABins("fakeR03LooseCN_"+lname,"03","rhoCentralNeutral","EAv1CN",0.5,selbkg,[0,1.2,2.4],vtxbins,[0.0,1.019])
            #tester.plotEffEtaEABins( "effR03Tight_"  +lname,"03","rho",              "EAv1",  0.1,seleff,[0,1.2,2.4],vtxbins,[0.7,1.019])
            tester.plotEffEtaEABins("fakeR03Tight_"  +lname,"03","rho",              "EAv1",  0.1,selbkg,[0,1.2,2.4],vtxbins,[0.0,0.319])
            #tester.plotEffEtaEABins( "effR03TightCN_"+lname,"03","rhoCentralNeutral","EAv1CN",0.1,seleff,[0,1.2,2.4],vtxbins,[0.7,1.019])
            #tester.plotEffEtaEABins("fakeR03TightCN_"+lname,"03","rhoCentralNeutral","EAv1CN",0.1,selbkg,[0,1.2,2.4],vtxbins,[0.0,0.319])
            #tester.plotROCEtaEABins("rocR03_"  +lname,"03","rho",              "EAv1",  selsig,selbkg,[0,1.2,2.4])
            #tester.plotROCEtaEABins("rocR03CN_"+lname,"03","rhoCentralNeutral","EAv1CN",selsig,selbkg,[0,1.2,2.4])

