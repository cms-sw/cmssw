#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from os.path import dirname
from CMGTools.TTHAnalysis.tools.plotDecorations import *

def replicaPdfBand(file, base, pdf, reference, eigenvectors, norm=False, relative=True):
    href = file.Get(base+reference)
    if not bool(href): return None
    bins = href.GetNbinsX()
    refnorm = href.Integral()
    values = [ [0 for b in xrange(bins) ] for r in xrange(eigenvectors+1) ]
    for e in xrange(eigenvectors+1):
        hist = file.Get("%s_%s_%d" % (base,pdf,e))
        if norm: hist.Scale(refnorm/hist.Integral())
        for b in xrange(bins):
            val  = hist.GetBinContent(b+1)
            vref = href.GetBinContent(b+1) if relative else 1
            values[e][b] = val/vref if vref else 1
    ret = ROOT.TGraphAsymmErrors(bins)
    for b in xrange(bins):
        avg = sum([values[i][b] for i in xrange(1,eigenvectors+1)])/eigenvectors
        rms = sqrt(sum([(values[i][b]-avg)**2  for i in xrange(1,eigenvectors+1)])/eigenvectors)
        ret.SetPoint(b, href.GetBinCenter(b+1), avg)
        dx = 0.5*href.GetBinWidth(b+1)
        ret.SetPointError(b, dx, dx, rms, rms)
    ret.SetName("%s_band_%s" % (base,pdf))
    ret.GetXaxis().SetTitle(href.GetXaxis().GetTitle())
    return ret

def eigenPdfBand(file, base, pdf, reference, eigenvectors, norm=False, relative=True):
    if (eigenvectors % 2 != 0): raise RuntimeError
    href = file.Get(base+reference)
    if not bool(href): return None
    bins = href.GetNbinsX()
    refnorm = href.Integral()
    values = [ [0,0,0]  for b in xrange(bins) ]
    central = file.Get("%s_%s_%d" % (base,pdf,0))
    if norm: central.Scale(refnorm/central.Integral())
    for b in xrange(bins):
        val  = central.GetBinContent(b+1)
        vref = href.GetBinContent(b+1) if relative else 1
        values[b][0] = val/vref if vref else 1
    for e in xrange(eigenvectors/2):
        h1 = file.Get("%s_%s_%d" % (base,pdf,2*e+1))
        h2 = file.Get("%s_%s_%d" % (base,pdf,2*e+2))
        if norm: h1.Scale(refnorm/h1.Integral())
        if norm: h2.Scale(refnorm/h2.Integral())
        for b in xrange(bins):
            vref = href.GetBinContent(b+1) if relative else 1
            if vref == 0: continue
            d1 = (h1.GetBinContent(b+1) - central.GetBinContent(b+1))/vref
            d2 = (h2.GetBinContent(b+1) - central.GetBinContent(b+1))/vref
            dlo = min([0,d1,d2])
            dhi = max([0,d1,d2])
            values[b][1] += dlo**2
            values[b][2] += dhi**2
    ret = ROOT.TGraphAsymmErrors(bins)
    for b in xrange(bins):
        ret.SetPoint(b, href.GetBinCenter(b+1), values[b][0])
        dx = 0.5*href.GetBinWidth(b+1)
        ret.SetPointError(b, dx, dx, sqrt(values[b][1]), sqrt(values[b][2]))
    ret.SetName("%s_band_%s" % (base,pdf))
    ret.GetXaxis().SetTitle(href.GetXaxis().GetTitle())
    return ret
    
def pdf4LHCEnv(*bands): 
    midpoint=False
    ret = bands[0].Clone()
    n = ret.GetN()
    for b in bands:
        if b.GetN() != n: 
            raise RuntimeError, "Band %s has %d entries, unlike reference who has %d" % (b.GetName(), b.GetN(), bands[0].GetName(), n)
    for i in xrange(n):
        hi = max([b.GetY()[i]+b.GetErrorYhigh(i) for b in bands])
        lo = min([b.GetY()[i]+b.GetErrorYhigh(i) for b in bands])
        mid = 0.5*(hi+lo) if midpoint else ret.GetY()[i]
        ret.SetPoint(i, ret.GetX()[i], mid)
        ret.SetPointError(i, ret.GetErrorXlow(i), ret.GetErrorYlow(i), mid-lo, hi-mid)
    return ret
if __name__ == "__main__":
    from sys import argv
    fin = ROOT.TFile(argv[1])
    fbase  = dirname(argv[1])
    fout = ROOT.TFile(fbase+"/systPDF.bands.root", "RECREATE")
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc")
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    c1 = ROOT.TCanvas("c1","c1")
    plots = [ "nJet25" ]
    if "3l" in argv[1]:
        plots += ["jet1pT","htJet25","htJet25ratio1224Lep","lepEta3","minDrllAFOS","bestMTopHad","finalMVA"]
    else:
        plots += ["lep2Pt" ,"lep2Eta" ,"drl2j" ,"mtW1" ,"htJet25" ,"mhtJet25","MVA_2LSS_4j_6var" ]
    for var in plots:
      #for L,N in ('', False), ('_norm',True):
      (L,N) = ('_norm',True)
      for P in "ttH TTW TTZ WZ ZZ".split():
        bands = {}
        for LR,R in ('', False), ('_ratio',True):
            bandN = replicaPdfBand(fin, var+"_"+P, "NNPDF21_100", "_CT10_0", 100, norm=N, relative=R)
            bandC = eigenPdfBand(fin, var+"_"+P, "CT10", "_CT10_0", 52, norm=N, relative=R)
            bandM = eigenPdfBand(fin, var+"_"+P, "MSTW2008lo68cl", "_CT10_0", 38, norm=N, relative=R)
            if not bandN: continue
            bandC.SetFillColor(33);  bandN.SetFillStyle(1001);
            bandN.SetFillColor(206); bandN.SetFillStyle(3006);
            bandM.SetFillColor(214); bandM.SetFillStyle(3007);
            bands["N"+LR] = bandN
            bands["M"+LR] = bandM
            bands["C"+LR] = bandC
            bandN.SetName("%s_%s_bands_%s%s_N" %(var,P,L,LR)); fout.WriteTObject(bandN)
            bandM.SetName("%s_%s_bands_%s%s_M" %(var,P,L,LR)); fout.WriteTObject(bandM)
            bandC.SetName("%s_%s_bands_%s%s_C" %(var,P,L,LR)); fout.WriteTObject(bandC)
            # make also PDF4LHC envelop
            bandE = pdf4LHCEnv(bandC,bandN,bandM)
            bandE.SetName("%s_%s_bands_%s%s" %(var,P,L,LR)); fout.WriteTObject(bandE)
            bands["E"+LR] = bandE
        if len(bands) != 8: continue
        bandN = bands["C"]
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
        frame.GetYaxis().SetRangeUser(0,1.4*ymax)
        frame.GetXaxis().SetLabelOffset(999) ## send them away
        frame.GetXaxis().SetTitleOffset(999) ## in outer space
        frame.GetYaxis().SetLabelSize(0.05)
        frame.GetYaxis().SetLabelSize(0.05)
        frame.GetYaxis().SetDecimals(True)
        frame.GetYaxis().SetTitle("Event yield")
        frame.Draw()
        for X in "CNM": 
            bands[X].Draw("E2 SAME")
        frame.Draw("AXIS SAME")
        ## Draw relaive prediction in the bottom frame
        p2.cd() 
        rframe = ROOT.TH1F("rframe","rframe",1,xmin,xmax)
        rframe.GetXaxis().SetTitle(bandN.GetXaxis().GetTitle())
        rframe.GetYaxis().SetRangeUser(0.88 if N else 0.70, 1.17 if N and (P != "TTW" or "el" not in argv[1]) else 1.35);
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
        for X in "CNM": 
            bands[X+"_ratio"].Draw("E2 SAME")
        line = ROOT.TLine(xmin,1,xmax,1)
        line.SetLineWidth(2);
        line.SetLineStyle(7);
        line.SetLineColor(1);
        line.Draw("L")
        rframe.Draw("AXIS SAME")
        p1.cd()
        leg = doLegend(.64,.73,.92,.91, textSize=0.05)
        leg.AddEntry(bands["C"], "CT10",     "F")
        leg.AddEntry(bands["N"], "NNPDF21",  "F")
        leg.AddEntry(bands["M"], "MSTW2008", "F")
        leg.Draw()
        c1.cd()
        doCMSSpam("CMS Simulation",textSize=0.035)
        c1.Print(fbase+"/systPDF_"+P+"_"+var+".png")
        c1.Print(fbase+"/systPDF_"+P+"_"+var+".pdf")
        del leg
        del frame
        del rframe
        del c1
    fout.Close()

