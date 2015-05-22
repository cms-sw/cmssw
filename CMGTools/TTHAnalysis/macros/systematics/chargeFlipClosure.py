
#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from CMGTools.TTHAnalysis.tools.plotDecorations import *


def haddThem(tdir, var, procs, norm=False):
    ret = None
    for P in procs.split():
        hist = tdir.Get("%s_%s" % (var,P))
        if hist == None: continue
        if ret == None: 
            ret = hist.Clone("%s_fakes" % var) 
            ret.SetDirectory(None)
        else:
            ret.Add(hist)
    if ret == None: 
        print "missing %s_%s in %s" % (var,P,tdir.GetName())
        tdir.ls()
        raise RuntimeError
    if norm and ret.Integral() > 0: 
       ret.Scale(1.0/ret.Integral("width")) 
    return ret

if __name__ == "__main__":
    from sys import argv
    fmc = ROOT.TFile("%sMCBG/%s.root" % (argv[1], argv[2]))
    fdd = ROOT.TFile("%sMCDD/%s.root" % (argv[1], argv[2]))
    fbase = "%sClosure/" % argv[1];
    ROOT.gSystem.Exec("mkdir -p %s" % fbase)
    #fout = ROOT.TFile(argv[3], "RECREATE")
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc")
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    c1 = ROOT.TCanvas("c1", "c1", 600, 750); c1.Draw()
    c1.SetWindowSize(600 + (600 - c1.GetWw()), (750 + (750 - c1.GetWh())));
    p1 = ROOT.TPad("pad1","pad1",0,0.31,1,1);
    p1.SetBottomMargin(0);
    p1.Draw();
    p2 = ROOT.TPad("pad2","pad2",0,0,1,0.31);
    p2.SetTopMargin(0);
    p2.SetBottomMargin(0.3);
    p2.SetFillStyle(0);
    p2.Draw();
    p1.cd();
    plots = [ ]
    if "3l" in argv[1]:
        plots = [(x,0,999) for x in ["nJet25","jet1pT","htJet25","htJet25ratio1224Lep","lepEta3","minDrllAFOS","bestMTopHad","finalMVA"]]
    else:
        plots = [("lep2Pt",0,999) ,("lep2Eta",0,999) ,("drl2j",0,999) ,("mtW1",0,999) ,("htJet25",0,999) ,("mhtJet25",0,999),("MVA_2LSS_4j_6var",0,999) ,("nJet25",0,999) ]
    print plots
    for var,rebin,maxbins in plots:
        for L,N in ('', False), ('_norm',True):
            hdd = haddThem(fdd, var, "QF_tt", norm=N)
            hmc = haddThem(fmc, var, "TTl", norm=N)
            if rebin: 
                hdd.Rebin(rebin)
                hmc.Rebin(rebin)
            hdd.SetFillColor(64)
            hdd.SetMarkerColor(64)
            hdd.SetMarkerStyle(1)
            hmc.SetLineWidth(2)
            unity = hdd.Clone()
            ratio = hmc.Clone()
            ratio.Divide(hdd)
            c2, ndf = 0,0
            for b in xrange(1,unity.GetNbinsX()+1):
                c = unity.GetBinContent(b)
                unity.SetBinError(b, unity.GetBinError(b)/c if c else 0)
                unity.SetBinContent(b, 1.0 if c else 0)
                if c and ratio.GetBinContent(b) != 0:
                    res = (ratio.GetBinContent(b)-1)/hypot(ratio.GetBinError(b), unity.GetBinError(b))
                    #print "bin %d: ratio %.1f +/- %.1f (ratio) +/- %.1f (ref): residual %.1f" % (
                    #            b, ratio.GetBinContent(b), ratio.GetBinError(b), unity.GetBinError(b), res)
                    if fabs(res) < 5 and b <= maxbins: 
                        c2 += res**2; ndf += 1
            if N: ndf -= 1
            unity.GetYaxis().SetRangeUser(0,2);
            unity.GetXaxis().SetTitleSize(0.14)
            unity.GetYaxis().SetTitleSize(0.14)
            unity.GetXaxis().SetLabelSize(0.11)
            unity.GetYaxis().SetLabelSize(0.11)
            unity.GetYaxis().SetNdivisions(505)
            unity.GetYaxis().SetDecimals(True)
            unity.GetYaxis().SetTitle("DD/MC.")
            unity.GetYaxis().SetTitleOffset(0.52);
            hdd.GetXaxis().SetLabelOffset(999) ## send them away
            hdd.GetXaxis().SetTitleOffset(999) ## in outer space
            hdd.GetYaxis().SetLabelSize(0.05)
            if N: hdd.GetYaxis().SetTitle("Event (normalized)")
            p1.cd()
            hdd.Draw("E2")
            hmc.Draw("E SAME")
            hdd.SetMaximum(1.5*max(hdd.GetMaximum(), hmc.GetMaximum()))
            hdd.SetMinimum(0)
            p2.cd()
            unity.Draw("E2");
            fitTGraph(ratio,1)
            unity.Draw("E2 SAME");
            unity.Draw("AXIS SAME");
            line = ROOT.TLine(unity.GetXaxis().GetXmin(),1,unity.GetXaxis().GetXmax(),1)
            line.SetLineWidth(2);
            line.SetLineColor(58);
            line.Draw("L")
            ratio.Draw("E SAME");
            p1.cd()
            doSpam("#chi^{2}/n = %.1f/%d (p-val %.3f)" % (c2,ndf,ROOT.TMath.Prob(c2,ndf)), 0.20,0.82,0.50,0.92, textSize=0.05)
            c1.Print(fbase+var+L+".png")
            c1.Print(fbase+var+L+".pdf")
            #exit()
    #fout.Close()

