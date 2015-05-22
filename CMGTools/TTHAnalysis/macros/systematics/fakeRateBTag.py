
#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from CMGTools.TTHAnalysis.tools.plotDecorations import *


def getThem(tdirs, var, procs, norm=False):
    ret = []
    norm0 = 0
    for T,P in zip(tdirs,procs):
        if not T.Get("%s_%s" % (var,P)):
            print "Missing %s_%s in %s" % (var,P,T.GetName())
            T.ls()
            exit()
        hist = T.Get("%s_%s" % (var,P)).Clone() 
        hist.SetDirectory(None)
        if norm: 
           if norm0 == 0: norm0 = hist.Integral("width")
           hist.Scale(norm0/hist.Integral("width")) 
        ret.append(hist)
    return ret

def makeRatios(hists):
    ref = hists[1]
    ratios = []
    for h in hists:
        r = h.Clone(h.GetName()+"_ratio")
        for b in xrange(1,ref.GetNbinsX()+1):
            if ref.GetBinContent(b) == 0: continue
            r.SetBinContent(b, r.GetBinContent(b)/ref.GetBinContent(b))
            r.SetBinError(b,   r.GetBinError(b)/ref.GetBinContent(b))
        ratios.append(r)
    return ratios

if __name__ == "__main__":
    from sys import argv
    frprocs = [ 'FR_data', 'FR_data_FRmc', 'FR_data_FRmc3j2b', 'FR_data_FRmc3j2B' ]
    colors  = [     1,            4,            209,                 2            ]
    files = [ ROOT.TFile("%s/%s/%s.root" % (argv[1], f, argv[2])) for f in frprocs ]
    fbase = "%s/Comparison/" % argv[1];
    ROOT.gSystem.Exec("mkdir -p %s" % fbase)
    ROOT.gSystem.Exec("cp /afs/cern.ch/user/g/gpetrucc/php/index.php %s" % fbase)
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
    plots = [ "nJet25"  ]
    if "3l" in argv[1]:
        plots += ["finalMVA"]
    else:
        plots += ["MVA_2LSS_4j_6var"]
    for var in plots:
        for L,N in ('', False), ('_norm',True):
            histos = getThem(files, var, frprocs, norm=N)
            for i,h in enumerate(histos): 
                h.SetLineWidth(2)
                h.SetLineColor(colors[i])
            ratios = makeRatios(histos) 
            ymax = max([h.GetMaximum() for h in histos])
            xmin = histos[0].GetXaxis().GetXmin()            
            xmax = histos[0].GetXaxis().GetXmax()
            if var == "nJet25": xmin = 1.5 if "3l" in argv[1] else 3.5
            ## Prepare split screen
            c1 = ROOT.TCanvas("c1", "c1", 600, 750); c1.Draw()
            c1.SetWindowSize(600 + (600 - c1.GetWw()), (750 + (750 - c1.GetWh())));
            p1 = ROOT.TPad("pad1","pad1",0,0.31,1,0.99);
            p1.SetBottomMargin(0);
            p1.Draw();
            p2 = ROOT.TPad("pad2","pad2",0,0,1,0.31);
            p2.SetTopMargin(0);
            p2.SetBottomMargin(0.3);
            p2.SetFillStyle(0);
            p2.Draw();
            p1.cd();
            ## Draw absolute prediction in top frame
            frame = ROOT.TH1F("frame","frame",1,xmin,xmax)
            frame.GetYaxis().SetRangeUser(0,1.6*ymax)
            frame.GetXaxis().SetLabelOffset(999) ## send them away
            frame.GetXaxis().SetTitleOffset(999) ## in outer space
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetDecimals(True)
            frame.GetYaxis().SetTitle("Event yield")
            frame.Draw()
            for X in histos[1:]: 
                X.Draw("HIST SAME")
            histos[0].Draw("E SAME")
            frame.Draw("AXIS SAME")
            ## Draw relaive prediction in the bottom frame
            p2.cd() 
            rframe = ROOT.TH1F("rframe","rframe",1,xmin,xmax)
            rframe.GetXaxis().SetTitle(histos[0].GetXaxis().GetTitle())
            rframe.GetYaxis().SetRangeUser(0.25,1.85 if N else 3.85);
            rframe.GetXaxis().SetTitleSize(0.14)
            rframe.GetYaxis().SetTitleSize(0.14)
            rframe.GetXaxis().SetLabelSize(0.11)
            rframe.GetYaxis().SetLabelSize(0.11)
            rframe.GetXaxis().SetNdivisions(505 if var == "nJet25" else 510)
            rframe.GetYaxis().SetNdivisions(505)
            rframe.GetYaxis().SetDecimals(True)
            rframe.GetYaxis().SetTitle("Ratio")
            rframe.GetYaxis().SetTitleOffset(0.52);
            rframe.Draw()
            ratios[1].SetFillColor(ROOT.kAzure-9)
            ratios[1].Draw("E2 SAME")
            for X in [ratios[0]]+ratios[2:]: 
                X.Draw("HIST SAME")
            line = ROOT.TLine(xmin,1,xmax,1)
            line.SetLineWidth(2);
            line.SetLineStyle(7);
            line.SetLineColor(1);
            line.Draw("L")
            rframe.Draw("AXIS SAME")
            p1.cd()
            leg = doLegend(.54,.73,.92,.91, textSize=0.05)
            leg.AddEntry(histos[0], "Data",       "L")
            leg.AddEntry(histos[1], "MC incl.",   "L")
            leg.AddEntry(histos[2], "MC b loose", "L")
            leg.AddEntry(histos[3], "MC b tight", "L")
            leg.Draw()
            c1.cd()
            doCMSSpam("CMS Preliminary",textSize=0.035)
            c1.Print(fbase+"/compFR_"+var+L+".png")
            c1.Print(fbase+"/compFR_"+var+L+".pdf")
            del leg
            del frame
            del rframe
            del c1

