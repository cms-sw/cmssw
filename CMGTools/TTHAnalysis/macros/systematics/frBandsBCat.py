
#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from os.path import dirname
from CMGTools.TTHAnalysis.tools.plotDecorations import *

def makeBand(file, base, norm=False, ratio=False, sources=[], ymin=0.02, basePostfix="",outPostfix=""):
    href = file.Get(base+basePostfix)
    if not bool(href): return None
    bins = href.GetNbinsX()
    fullscale = href.Integral()
    allsources = [ s+"Up"  for s in sources ] + [ s+"Dn" for s in sources ]
    hs = [ file.Get("%s_%s" % (base,s)) for s in allsources ]
    for (h,s) in zip(hs,allsources):
        if not h: raise RuntimeError, "Missing histogram %s_%s" % (base,s)
    if norm:
        for h in hs: h.Scale(fullscale/h.Integral())
    ret = ROOT.TGraphAsymmErrors(bins)
    for b in xrange(bins):
        y0 = href.GetBinContent(b+1)
        if y0/fullscale < ymin: continue
        ys = [ h.GetBinContent(b+1) for h in hs ]
        yhi = max(ys); ylo = min(ys)
        ymid = 0.5*(yhi+ylo)
        dx = 0.5*href.GetBinWidth(b+1)
        div = y0 if ratio else 1.0
        ret.SetPoint(b, href.GetBinCenter(b+1), ymid/div)
        ret.SetPointError(b, dx, dx, (ymid-ylo)/div, (yhi-ymid)/div)
        #print "bin %d, x = %5.2f: ref = %5.2f, min = %5.2f, max = %5.2f, ratio %s: point = %6.3f   -%6.3f / +%6.3f" % (b+1,href.GetBinCenter(b+1),y0,ylo,yhi,ratio,ymid/div,(ymid-ylo)/div, (yhi-ymid)/div)
    ret.SetName("%s%s%s_band%s" % (base,("_ratio" if ratio else ""),("_norm" if norm else ""),outPostfix))
    ret.GetXaxis().SetTitle(href.GetXaxis().GetTitle())
    return ret

 
if __name__ == "__main__":
    from sys import argv
    fin = ROOT.TFile(argv[1])
    fbase  = dirname(argv[1])
    fout = ROOT.TFile(fbase+"/systFR.bands.root", "RECREATE")
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc")
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    plots = [ "nJet25" ]
    leptons = []
    if "3l" in argv[1] or "em" in argv[1]:
        leptons = [ "mu", "el" ]
    elif "mumu" in argv[1]:
        leptons = [ "mu" ]
    elif "ee" in argv[1]:
        leptons = [ "el" ]
    else: raise RuntimeError, "No idea which leptons to test"
    if "3l" in argv[1]:
        plots += [ "finalMVA" ]
    else:
        plots += [ "MVA_2LSS_4j_6var" ]

    for L in leptons:
        FRnorm = [ L ];
        FRshape = [ L+"BT" ]
        for var in plots:
            P = "FR_data"
            bands = {}
            for LR,R in ('', False), ('_ratio',True):
                bandN = makeBand(fin, var+"_"+P, norm=False, ratio=R, sources=FRnorm, outPostfix="_N")
                bandS = makeBand(fin, var+"_"+P, norm=True, ratio=R, sources=FRshape, outPostfix="_S")
                if not bandN or not bandS: continue
                bandN.SetFillColor(ROOT.kGreen-7 if L == "el" else ROOT.kAzure-9);   bandN.SetFillStyle(1001);
                bandS.SetFillColor(ROOT.kGreen+2 if L == "el" else ROOT.kAzure+2); bandS.SetFillStyle(1001);
                bandN.SetName("%s_%s_bands_%s_norm%s" %(var,P,L,LR)); fout.WriteTObject(bandN)
                bandS.SetName("%s_%s_bands_%s_shape%s"%(var,P,L,LR)); fout.WriteTObject(bandS)
                bands["norm" +LR] = bandN
                bands["shape"+LR] = bandS
            if len(bands) != 4: continue
            bandN = bands["norm"]
            bandN.Sort()
            xmin = bandN.GetX()[0]-bandN.GetErrorXlow(0)
            xmax = bandN.GetX()[bandN.GetN()-1]+bandN.GetErrorXhigh(bandN.GetN()-1)
            ymax = max([bandN.GetY()[i]+1.3*bandN.GetErrorYhigh(i) for i in xrange(bandN.GetN())])
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
            frame.GetYaxis().SetRangeUser(0,1.1*ymax)
            frame.GetXaxis().SetLabelOffset(999) ## send them away
            frame.GetXaxis().SetTitleOffset(999) ## in outer space
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetLabelSize(0.05)
            frame.GetYaxis().SetTitle("Event yield")
            frame.Draw()
            bands["norm" ].Draw("E2 SAME")
            bands["shape"].Draw("E2 SAME")
            frame.Draw("AXIS SAME")
            ## Draw relaive prediction in the bottom frame
            p2.cd() 
            rframe = ROOT.TH1F("rframe","rframe",1,xmin,xmax)
            rframe.GetXaxis().SetTitle(bandN.GetXaxis().GetTitle())
            rframe.GetYaxis().SetRangeUser(0,1.95);
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
            bands[ "norm_ratio"].Draw("E2 SAME")
            bands["shape_ratio"].Draw("E2 SAME")
            line = ROOT.TLine(xmin,1,xmax,1)
            line.SetLineWidth(2);
            line.SetLineStyle(7);
            line.SetLineColor(1);
            line.Draw("L")
            rframe.Draw("AXIS SAME")
            p1.cd()
            leg = doLegend(.67,.75,.92,.91, textSize=0.05)
            leg.AddEntry(bandN, "Norm.",  "F")
            leg.AddEntry(bandS, "Shape",  "F")
            leg.Draw()
            c1.cd()
            doCMSSpam("CMS Preliminary",textSize=0.035)
            c1.Print(fbase+"/systFR_"+L+"_"+var+".png")
            c1.Print(fbase+"/systFR_"+L+"_"+var+".pdf")
            del leg
            del frame
            del rframe
            del c1
    fout.Close()

