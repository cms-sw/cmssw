
#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from os.path import dirname
from CMGTools.TTHAnalysis.tools.plotDecorations import *

def scaleBand(file, base, center=False, norm=False, scales=["scaleUp","scaleDown"], ymin=0.05, basePostfix="", outPostfix="", relative=True, rebase=True):
    href = file.Get(base+basePostfix)
    if not bool(href): return None
    print "doing ",base," with ",scales
    bins = href.GetNbinsX()
    fullscale = href.Integral()
    hs = [ file.Get("%s_%s" % (base,s)) for s in scales ]
    for (h,s) in zip(hs,scales):
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
        if center: 
            if relative:
                y0 = ymid
            else:
                yhi += (y0-ymid)
                ylo += (y0-ymid)
                ymid = y0
                y0 = 1
        elif not relative: 
            y0 = 1
        ret.SetPoint(b, href.GetBinCenter(b+1), ymid/y0)
        ret.SetPointError(b, dx, dx, (ymid-ylo)/y0, (yhi-ymid)/y0)
    ret.SetName("%s_band%s" % (base,outPostfix))
    ret.GetXaxis().SetTitle(href.GetXaxis().GetTitle())
    return ret

def qsumBands(*bands):
    ret = bands[0].Clone()
    for i in xrange(ret.GetN()):
        dydn = sqrt(sum([ b.GetErrorYlow(i)**2  for b in bands])) 
        dyup = sqrt(sum([ b.GetErrorYhigh(i)**2 for b in bands])) 
        ret.SetPointError(i,ret.GetErrorXlow(i),ret.GetErrorXhigh(i),dydn,dyup)
    return ret

if __name__ == "__main__":
    from sys import argv
    fin = ROOT.TFile(argv[1])
    fbase  = dirname(argv[1])
    fout = ROOT.TFile(fbase+"/systTH.bands.root", "RECREATE")
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc")
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    plots = [ "nJet25" ]
    if "3l" in argv[1]:
        plots += ["jet1pT","htJet25","htJet25ratio1224Lep","lepEta3","minDrllAFOS","bestMTopHad","finalMVA"]
    else:
        plots += ["lep2Pt" ,"lep2Eta" ,"drl2j" ,"mtW1" ,"htJet25" ,"mhtJet25","MVA_2LSS_4j_6var" ]
    for var in plots:
        #for L,N in ('', False), ('_norm',True):
        (L,N) = ('_norm',True)
        for P in "TTH TTW TTZ".split():
            scales  = [ "scaleUp", "scaleDown", "nominal" ]
            matches = [ "xqtUp", "xqtDown", "nominal" ]
            bands = {}
            for LR,R in ('', False), ('_ratio',True):
                bandS = scaleBand(fin, var+"_"+P, True, norm=N, scales=scales, ymin=1e-3, relative=R)
                bandM = scaleBand(fin, var+"_"+P, True, norm=N, scales=matches, ymin=1e-3, relative=R)
                if not bandS: continue
                bandQ = qsumBands(bandS,bandM)
                bandQ.SetFillColor(ROOT.kAzure-9);  bandS.SetFillStyle(1001);
                bandS.SetFillColor(ROOT.kAzure+2); bandS.SetFillStyle(1001);
                bandM.SetFillColor(1);   bandM.SetFillStyle(3004); 
                bands["Q"+LR] = bandQ
                bands["S"+LR] = bandS
                bands["M"+LR] = bandM
                bandQ.SetName("%s_%s_bands_%s%s" %(var,P,L,LR)); fout.WriteTObject(bandQ)
                bandS.SetName("%s_%s_bands_%s%s_S" %(var,P,L,LR)); fout.WriteTObject(bandS)
                bandM.SetName("%s_%s_bands_%s%s_M" %(var,P,L,LR)); fout.WriteTObject(bandM)
            if len(bands) != 6: continue
            bandN = bands["Q"]
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
            for X in "QSM": 
                bands[X].Draw("E2 SAME")
            frame.Draw("AXIS SAME")
            ## Draw relaive prediction in the bottom frame
            p2.cd() 
            rframe = ROOT.TH1F("rframe","rframe",1,xmin,xmax)
            rframe.GetXaxis().SetTitle(bandN.GetXaxis().GetTitle())
            rframe.GetYaxis().SetRangeUser(0.25,1.85);
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
            for X in "QSM": 
                bands[X+"_ratio"].Draw("E2 SAME")
            line = ROOT.TLine(xmin,1,xmax,1)
            line.SetLineWidth(2);
            line.SetLineStyle(7);
            line.SetLineColor(1);
            line.Draw("L")
            rframe.Draw("AXIS SAME")
            p1.cd()
            leg = doLegend(.64,.73,.92,.91, textSize=0.05)
            leg.AddEntry(bands["S"], "Scale",    "F")
            leg.AddEntry(bands["M"], "Matching", "F")
            leg.AddEntry(bands["Q"], "Combined", "F")
            leg.Draw()
            c1.cd()
            doCMSSpam("CMS Simulation",textSize=0.035)
            c1.Print(fbase+"/systTH_"+P+"_"+var+".png")
            c1.Print(fbase+"/systTH_"+P+"_"+var+".pdf")
            del leg
            del frame
            del rframe
            del c1
    fout.Close()

