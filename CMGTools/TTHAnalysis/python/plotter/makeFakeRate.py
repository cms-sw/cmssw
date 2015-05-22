from math import *
from os.path import basename

import sys
sys.argv.append('-b-')
import ROOT
ROOT.gROOT.SetBatch(True)
sys.argv.remove('-b-')

from CMGTools.TTHAnalysis.plotter.mcPlots import *

def H1DToH2D(h1d,h2d,func):
    ax = h2d.GetXaxis()
    ay = h2d.GetYaxis()
    a1 = h1d.GetXaxis()
    for bx in xrange(1,h2d.GetNbinsX()+1):
       x = ax.GetBinCenter(bx) 
       for by in xrange(1,h2d.GetNbinsY()+1):
           y = ay.GetBinCenter(by) 
           ib = a1.FindBin(func(x,y))
           h2d.SetBinContent(bx, by, h1d.GetBinContent(ib))
           h2d.SetBinError(bx, by, h1d.GetBinError(ib))

class MCFakeRate:
    def __init__(self,plotFileName, denDir, numDir, options):
        self._plotFileName = plotFileName
        self._denDir = denDir
        self._numDir = numDir
        self._plots = PlotFile(plotFileName,options)
        self._numFile = ROOT.TFile.Open(self._numDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._denFile = ROOT.TFile.Open(self._denDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._options = options
        self._outfile = ROOT.TFile.Open(self._denDir+"/"+options.out, "RECREATE")
    def makePlotsBySource(self,mca):
        for p in self._plots.plots():
            asig = (mca.listSignals()+mca.listBackgrounds())[0]
            sig  = [self._numFile.Get(p.name + "_"+asig).Clone("snum"), self._denFile.Get(p.name + "_"+asig).Clone("sden")]
            for i in 0,1: sig[i].Reset(); 
            for proc in mca.listSignals()+mca.listBackgrounds():
                if self._numFile.Get(p.name + "_" + proc):
                    sig[0].Add(self._numFile.Get(p.name + "_" + proc))
                    sig[1].Add(self._denFile.Get(p.name + "_" + proc))
            if "TH1" in sig[0].ClassName():
                    h = sig
                    text = "    Fake rate vs %s\n" % (p.name)
                    text += "%3s   %8s  %8s    %6s +/- %6s\n" % ("bin", "xmin ", "xmax ", "value", "error ")
                    text += "%3s   %8s  %8s    %6s-----%6s\n" % ("---", "------", "------", "------", "------")
                    for b in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                        f = n/float(d) if d > 0 else 0; 
                        wavg = (h[0].GetBinError(b)**2) /h[0].GetBinError(b) if h[0].GetBinError(b) else 1
                        df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                        text += "%3d   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (b, h[0].GetXaxis().GetBinLowEdge(b),h[0].GetXaxis().GetBinUpEdge(b), f,df)
                        h[0].SetBinContent(b, f) 
                        h[0].SetBinError(b, df)
                    c1 = ROOT.TCanvas("FR_"+p.name, p.name, 600, 400)
                    h[0].GetYaxis().SetTitle("Fake rate");
                    h[0].GetYaxis().SetRangeUser(0.0,0.4 if self._options.maxRatioRange[1] > 1 else self._options.maxRatioRange[1])
                    h[0].SetLineColor(ROOT.kRed)
                    h[0].SetMarkerColor(ROOT.kRed)
                    h[0].SetLineWidth(2)
                    h[0].Draw("E1")
                    ROOT.gStyle.SetErrorX(0.5)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s.%s" % (self._denDir, p.name, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s.%s" % (self._denDir, p.name, ext))
                    h[0].SetName("FR_%s" % (p.name)); self._outfile.WriteTObject(h[0]);

class FakeRateSimple:
    def __init__(self,plotFileName, denDir, numDir, options):
        self._plotFileName = plotFileName
        self._denDir = denDir
        self._numDir = numDir
        self._plots = PlotFile(plotFileName,options)
        self._numFile = ROOT.TFile.Open(self._numDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._denFile = ROOT.TFile.Open(self._denDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._options = options
        self._outfile = ROOT.TFile.Open(self._denDir+"/"+options.out, "RECREATE")
    def makePlotsBySource(self,mca):
        for p in self._plots.plots():
            asig = mca.listSignals()[0]
            abkg = mca.listBackgrounds()[0]
            data = [self._numFile.Get(p.name + "_data"), self._denFile.Get(p.name + "_data")]
            sig  = [self._numFile.Get(p.name + "_"+asig).Clone("snum"), self._denFile.Get(p.name + "_"+asig).Clone("sden")]
            bkg  = [self._numFile.Get(p.name + "_"+abkg).Clone("bnum"), self._denFile.Get(p.name + "_"+abkg).Clone("bden")]
            if not data[0]: continue
            for i in 0,1:
                sig[i].Reset(); bkg[i].Reset();
            for proc in mca.listSignals():
                if self._numFile.Get(p.name + "_" + proc):
                    sig[0].Add(self._numFile.Get(p.name + "_" + proc))
                    sig[1].Add(self._denFile.Get(p.name + "_" + proc))
            for proc in mca.listBackgrounds():
                if self._numFile.Get(p.name + "_" + proc):
                    bkg[0].Add(self._numFile.Get(p.name + "_" + proc))
                    bkg[1].Add(self._denFile.Get(p.name + "_" + proc))
            mc = [ sig[0].Clone("mcnum"), sig[1].Clone("mcden") ]
            for i in 0,1 : mc[i].Add(bkg[i])
            if "TH1" in data[0].ClassName():
                color = { 'data':1, 'qcd':ROOT.kOrange+10, 'ewk':ROOT.kCyan+2, 'mc':ROOT.kOrange+4 }
                for l,h in ('data',data),('qcd',sig),('ewk',bkg),('mc',mc):
                    text = "    Fake rate vs %s for %s\n" % (p.name,l)
                    text += "%3s   %8s  %8s    %6s +/- %6s\n" % ("bin", "xmin ", "xmax ", "value", "error ")
                    text += "%3s   %8s  %8s    %6s-----%6s\n" % ("---", "------", "------", "------", "------")
                    for b in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                        f = n/float(d) if d > 0 else 0; 
                        if l == "data":
                            df = sqrt(f*(1-f)/d) if d > 0 else 0
                        else:
                            # get average weight of events (at numerator)
                            wavg = (h[0].GetBinError(b)**2) /h[0].GetBinError(b) if h[0].GetBinError(b) else 1
                            df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                        text += "%3d   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (b, h[0].GetXaxis().GetBinLowEdge(b),h[0].GetXaxis().GetBinUpEdge(b), f,df)
                        h[0].SetBinContent(b, f) 
                        h[0].SetBinError(b, df)
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 600, 400)
                    h[0].GetYaxis().SetTitle("Fake rate");
                    h[0].SetLineColor(color[l])
                    h[0].SetMarkerColor(color[l])
                    if l != "ewk": h[0].GetYaxis().SetRangeUser(0.0,0.4 if self._options.maxRatioRange[1] > 1 else self._options.maxRatioRange[1]);
                    else:          h[0].GetYaxis().SetRangeUser(0.8,1.0);
                    h[0].SetLineWidth(2)
                    h[0].Draw("E1")
                    ROOT.gStyle.SetErrorX(0.5)
                    #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext))
                    h[0].SetName("FR_%s_%s" % (p.name,l)); self._outfile.WriteTObject(h[0]);
                c1 = ROOT.TCanvas("FR_"+p.name+"_stack", p.name, 600, 400)
                sig[0].Draw("E1")
                mc[0].Draw("E1 SAME")
                bkg[0].Draw("E1 SAME")
                data[0].Draw("E1 SAME")
                for ext in self._options.printPlots.split(","):
                    if ext == "txt": continue
                    c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, "stack", ext))
            elif "TH2" in data[0].ClassName():
                for l,h in ('data',data),('qcd',sig),('ewk',bkg),('mc',mc):
                    text = "    Fake rate vs %s for %s; xvar = %s, yvar = %s\n" % (p.name,l, p.getOption('XTitle','x'), p.getOption('YTitle','x'))
                    text += "%3s %3s   %8s  %8s   %8s  %8s    %6s +/- %6s\n" % ("bx", "by", "xmin ", "xmax ", "ymin ", "ymax ", "value", "error ")
                    text += "%3s %3s   %8s  %8s   %8s  %8s    %6s-----%6s\n" % ("---","---",  "------", "------", "------", "------", "------", "------")
                    for bx in xrange(1,h[0].GetNbinsX()+1):
                      for by in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(bx,by), h[1].GetBinContent(bx,by)
                        f = n/float(d) if d > 0 else 0; 
                        if l == "data":
                            df = sqrt(f*(1-f)/d) if d > 0 else 0
                        else:
                            # get average weight of events (at numerator)
                            wavg = (h[0].GetBinError(bx,by)**2) /h[0].GetBinError(bx,by) if h[0].GetBinError(bx,by) else 1
                            df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0 and f > 0 and f < 1 else 0
                        text += "%3d %3d   % 8.3f  % 8.3f   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (bx,by, h[0].GetXaxis().GetBinLowEdge(bx),h[0].GetXaxis().GetBinUpEdge(bx), h[0].GetYaxis().GetBinLowEdge(by),h[0].GetYaxis().GetBinUpEdge(by), f,df)
                        h[0].SetBinContent(bx,by, f) 
                        h[0].SetBinError(bx,by, df)
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 900, 800)
                    c1.SetRightMargin(0.20)
                    ROOT.gStyle.SetErrorX(0.5)
                    ROOT.gStyle.SetPaintTextFormat(".3f")
                    ROOT.gStyle.SetTextFont(62)
                    h[0].GetZaxis().SetTitle("Fake rate");
                    if l != "ewk": h[0].GetZaxis().SetRangeUser(0.0,0.4);
                    else:          h[0].GetZaxis().SetRangeUser(0.8,1.0);
                    h[0].Draw("COLZ TEXT90E")
                    h[0].SetMarkerSize(1.5)
                    #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext))
                    h[0].SetName("FR_%s_%s" % (p.name,l)); self._outfile.WriteTObject(h[0]);
            else:
                raise RuntimeError, "No idea how to handle a " + data[0].ClassName()

class FakeRateMET1Bin:
    def __init__(self,plotFileName, denDir, numDir, options):
        self._plotFileName = plotFileName
        self._denDir = denDir
        self._numDir = numDir
        self._plots = PlotFile(plotFileName,options)
        self._numFile = ROOT.TFile.Open(self._numDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._denFile = ROOT.TFile.Open(self._denDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._options = options
    def integral(self,h,xmin,xmax):
        n, n2 = 0, 0
        for b in xrange(1,h.GetNbinsX()+1):
            if (h.GetXaxis().GetBinCenter(b) > xmin and h.GetXaxis().GetBinCenter(b) < xmax):
                n  += h.GetBinContent(b)
                n2 += h.GetBinError(b)**2
        return [n, sqrt(n2)]
    def frFromRange(self,h,xmin,xmax):
        n = self.integral(h[0],xmin,xmax)
        d = self.integral(h[1],xmin,xmax)
        wavg = n[1]**2/n[0] if n[0] else 1
        f = n[0]/float(d[0]) if d[0] > 0 else 0
        df = sqrt(f*(1-f)/(d[0]/wavg)) if wavg > 0 and (d[0]/wavg) > 0  and f > 0 and f <  1 else 0
        return (f,df) 
    def rslp(self,hdata,hewk,low,high):
        data_s = self.integral(hdata[1],low[0],low[1])
        data_l = self.integral(hdata[1],high[0],high[1])
        ewk_s = self.integral(hewk[1],low[0],low[1])
        ewk_l = self.integral(hewk[1],high[0],high[1])
        if ewk_s[0] == 0 or data_l[0] == 0: return [0,0,0]
        if data_s[0] == 0 or ewk_l[0] == 0: return [-1,0,0]
        r = (data_l[0]/data_s[0]) / (ewk_l[0]/ewk_s[0]) 
        dr_stat = r * sqrt((ewk_l[1]/ewk_l[0])**2 + (ewk_s[1]/ewk_s[0])**2)
        # for syst, we take the variation of the EWK fake rate
        ewk_f_s = self.frFromRange(hewk,low[0],low[1])
        ewk_f_l = self.frFromRange(hewk,high[0],high[1])
        dr_syst = r * abs(ewk_f_s[0]-ewk_f_l[0]) 
        return [r, dr_stat, dr_syst]
    def makePlotsBySource(self,mca,pname="met"):
        for p in self._plots.plots():
            if p.name != pname: continue
            asig = mca.listSignals()[0]
            abkg = mca.listBackgrounds()[0]
            data = [self._numFile.Get(p.name + "_data"), self._denFile.Get(p.name + "_data")]
            if not data[0]: continue
            sig  = [self._numFile.Get(p.name + "_"+asig).Clone("snum"), self._denFile.Get(p.name + "_"+asig).Clone("sden")]
            bkg  = [self._numFile.Get(p.name + "_"+abkg).Clone("bnum"), self._denFile.Get(p.name + "_"+abkg).Clone("bden")]
            for i in 0,1:
                sig[i].Reset(); bkg[i].Reset();
            for proc in mca.listSignals():
                if self._numFile.Get(p.name + "_" + proc):
                    sig[0].Add(self._numFile.Get(p.name + "_" + proc))
                    sig[1].Add(self._denFile.Get(p.name + "_" + proc))
            for proc in mca.listBackgrounds():
                if self._numFile.Get(p.name + "_" + proc):
                    bkg[0].Add(self._numFile.Get(p.name + "_" + proc))
                    bkg[1].Add(self._denFile.Get(p.name + "_" + proc))
            mc = [ sig[0].Clone("mcnum"), sig[1].Clone("mcden") ]
            for i in 0,1 : mc[i].Add(bkg[i])
            met_s = (  0., 20.); met_l = ( 45., 80. )
            f_s   = self.frFromRange(data, met_s[0], met_s[1])
            f_l   = self.frFromRange(data, met_l[0], met_l[1])
            r_slp = self.rslp(data,bkg,met_s,met_l)
            f_qcd = (f_s[0] - f_l[0]*r_slp[0])/(1-r_slp[0])
            if self._options.globalRebin:
                for i in 0,1:
                    for s in data,sig,bkg,mc:
                        s[i].Rebin(self._options.globalRebin)
            if "TH1" in data[0].ClassName():
                color = { 'data':1, 'qcd':ROOT.kOrange+10, 'ewk':ROOT.kCyan+2, 'mc':ROOT.kOrange+4 }
                for l,h in ('data',data),('qcd',sig),('ewk',bkg),('mc',mc):
                    text = "    Fake rate vs %s for %s\n" % (p.name,l)
                    text += "%3s   %8s  %8s    %6s +/- %6s\n" % ("bin", "xmin ", "xmax ", "value", "error ")
                    text += "%3s   %8s  %8s    %6s-----%6s\n" % ("---", "------", "------", "------", "------")
                    for b in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                        f = n/float(d) if d > 0 else 0; 
                        if l == "data":
                            df = sqrt(f*(1-f)/d) if d > 0 else 0
                        else:
                            # get average weight of events (at numerator)
                            wavg = (h[0].GetBinError(b)**2) /h[0].GetBinContent(b) if h[0].GetBinError(b) else 1
                            df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                        text += "%3d   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (b, h[0].GetXaxis().GetBinLowEdge(b),h[0].GetXaxis().GetBinUpEdge(b), f,df)
                        h[0].SetBinContent(b, f) 
                        h[0].SetBinError(b, df)
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 600, 400)
                    h[0].GetYaxis().SetTitle("Fake rate");
                    h[0].SetLineColor(color[l])
                    h[0].SetMarkerColor(color[l])
                    if l != "ewk": h[0].GetYaxis().SetRangeUser(0.0,0.4 if self._options.maxRatioRange[1] > 1 else self._options.maxRatioRange[1]);
                    else:          h[0].GetYaxis().SetRangeUser(0.8,1.0);
                    h[0].SetLineWidth(2)
                    h[0].Draw("E1")
                    ROOT.gStyle.SetErrorX(0.5)
                    #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext))
                c1 = ROOT.TCanvas("FR_"+p.name+"_stack", p.name, 600, 400)
                sig[0].Draw("E1")
                mc[0].Draw("E1 SAME")
                bkg[0].Draw("E1 SAME")
                data[0].Draw("E1 SAME")
                lsub = ROOT.TLine(sig[0].GetXaxis().GetXmin(), f_qcd, sig[0].GetXaxis().GetXmax(), f_qcd);
                lsub.SetLineWidth(3) 
                lsub.SetLineColor(ROOT.kGreen+2) 
                lsub.Draw("SAME")
                for ext in self._options.printPlots.split(","):
                    if ext == "txt": continue
                    c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, "stack", ext))
            elif "TH2" in data[0].ClassName():
                for l,h in ('data',data),('qcd',sig),('ewk',bkg),('mc',mc):
                    text = "    Fake rate vs %s for %s; xvar = %s, yvar = %s\n" % (p.name,l, p.getOption('XTitle','x'), p.getOption('YTitle','x'))
                    text += "%3s %3s   %8s  %8s   %8s  %8s    %6s +/- %6s\n" % ("bx", "by", "xmin ", "xmax ", "ymin ", "ymax ", "value", "error ")
                    text += "%3s %3s   %8s  %8s   %8s  %8s    %6s-----%6s\n" % ("---","---",  "------", "------", "------", "------", "------", "------")
                    for bx in xrange(1,h[0].GetNbinsX()+1):
                      for by in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(bx,by), h[1].GetBinContent(bx,by)
                        f = n/float(d) if d > 0 else 0; 
                        if l == "data":
                            df = sqrt(f*(1-f)/d) if d > 0 else 0
                        else:
                            # get average weight of events (at numerator)
                            wavg = (h[0].GetBinError(bx,by)**2) /h[0].GetBinError(bx,by) if h[0].GetBinError(bx,by) else 1
                            df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0 and f > 0 and f < 1 else 0
                        text += "%3d %3d   % 8.3f  % 8.3f   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (bx,by, h[0].GetXaxis().GetBinLowEdge(bx),h[0].GetXaxis().GetBinUpEdge(bx), h[0].GetYaxis().GetBinLowEdge(by),h[0].GetYaxis().GetBinUpEdge(by), f,df)
                        h[0].SetBinContent(bx,by, f) 
                        h[0].SetBinError(bx,by, df)
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 900, 800)
                    c1.SetRightMargin(0.20)
                    ROOT.gStyle.SetErrorX(0.5)
                    ROOT.gStyle.SetPaintTextFormat(".3f")
                    ROOT.gStyle.SetTextFont(62)
                    h[0].GetZaxis().SetTitle("Fake rate");
                    if l != "ewk": h[0].GetZaxis().SetRangeUser(0.0,0.4);
                    else:          h[0].GetZaxis().SetRangeUser(0.8,1.0);
                    h[0].Draw("COLZ TEXT90E")
                    h[0].SetMarkerSize(1.5)
                    #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, l, ext))
            else:
                raise RuntimeError, "No idea how to handle a " + data[0].ClassName()

class FakeRateMET1D(FakeRateMET1Bin):
    def __init__(self, plotFileName, denDir, numDir, options):
        FakeRateMET1Bin.__init__(self, plotFileName, denDir, numDir, options)
    def fill1DY(self,h2d,h1d,ix):
        if type(h2d) == type([]):
            for h2di,h1di in zip(h2d,h1d):
                self.fill1DY(h2di,h1di,ix)
        else:
            for iy in xrange(1,h2d.GetNbinsY()+1):
                h1d.SetBinContent(iy, h2d.GetBinContent(ix,iy))
                h1d.SetBinError(iy, h2d.GetBinError(ix,iy))
    def fill1DX(self,h2d,h1d,iy):
        if type(h2d) == type([]):
            for h2di,h1di in zip(h2d,h1d):
                self.fill1DX(h2di,h1di,iy)
        else:
            for ix in xrange(1,h2d.GetNbinsX()+1):
                h1d.SetBinContent(ix, h2d.GetBinContent(ix,iy))
                h1d.SetBinError(ix, h2d.GetBinError(ix,iy))
    def makePlotsBySource(self,mca,xname="met"):
        for p in self._plots.plots():
            if not p.name.endswith("_"+xname): continue
            print p.name
            asig = mca.listSignals()[0]
            abkg = mca.listBackgrounds()[0]
            data = [self._numFile.Get(p.name + "_data").Clone("dnum"), self._denFile.Get(p.name + "_data").Clone("dden")]
            if not data[0]: continue
            sig  = [self._numFile.Get(p.name + "_"+asig).Clone("snum"), self._denFile.Get(p.name + "_"+asig).Clone("sden")]
            bkg  = [self._numFile.Get(p.name + "_"+abkg).Clone("bnum"), self._denFile.Get(p.name + "_"+abkg).Clone("bden")]
            for i in 0,1:
                sig[i].Reset(); bkg[i].Reset();
            for proc in mca.listSignals():
                if self._numFile.Get(p.name + "_" + proc):
                    sig[0].Add(self._numFile.Get(p.name + "_" + proc))
                    sig[1].Add(self._denFile.Get(p.name + "_" + proc))
            for proc in mca.listBackgrounds():
                if self._numFile.Get(p.name + "_" + proc):
                    bkg[0].Add(self._numFile.Get(p.name + "_" + proc))
                    bkg[1].Add(self._denFile.Get(p.name + "_" + proc))
            color = { 'data':1, 'qcd':ROOT.kOrange+10, 'ewk':ROOT.kCyan+2, 'mc':ROOT.kOrange+4, 'dsub':ROOT.kGreen+3 }
            # NOW WHAT??
            # step 0: prepare sutable plots for projections
            sigy =  [ h.ProjectionY() for h in sig  ]
            bkgy =  [ h.ProjectionY() for h in bkg  ]
            datay = [ h.ProjectionY() for h in data ]
            sigx =  [ h.ProjectionX() for h in sig  ]
            bkgx =  [ h.ProjectionX() for h in bkg  ]
            datax = [ h.ProjectionX() for h in data ]
            dcorrx = [ h.Clone("_corr") for h in datax ]
            # step 0: make uncorrected FR
            for l,h in ('data',datax),('qcd',sigx),('ewk',bkgx):
                for b in xrange(1,h[0].GetNbinsX()+1):
                    n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                    f = n/float(d) if d > 0 else 0; 
                    if l == "data":
                        df = sqrt(f*(1-f)/d) if d > 0 else 0
                    else:
                        # get average weight of events (at numerator)
                        wavg = (h[0].GetBinError(b)**2) /h[0].GetBinContent(b) if h[0].GetBinError(b) else 1
                        df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                    h[0].SetBinContent(b, f) 
                    h[0].SetBinError(b, df)
            # step 1: make plots of FR vs MET in each bin
            for ix in xrange(1,datax[0].GetNbinsX()+1):
                xlabel = "%s_%g_%g" % (p.name.replace("_"+xname,""),datax[0].GetXaxis().GetBinLowEdge(ix),datax[0].GetXaxis().GetBinUpEdge(ix))
                self.fill1DY(sig, sigy, ix) 
                self.fill1DY(bkg, bkgy, ix) 
                self.fill1DY(data,datay,ix) 
                met_s = (  0., 20.); met_l = ( 45., 80. )
                if self.integral(datay[1], met_s[0], met_l[1] )[0] == 0:
                    dcorrx[0].SetBinContent(ix, 0.0)
                    dcorrx[0].SetBinError(ix,   0.0) 
                    continue
                f_s   = self.frFromRange(datay, met_s[0], met_s[1])
                f_l   = self.frFromRange(datay, met_l[0], met_l[1])
                r_slp = self.rslp(datay,bkgy,met_s,met_l)
                if r_slp[0] == -1 or r_slp[0] >= 1:
                    f_qcd = 0 
                    dcorrx[0].SetBinContent(ix, 0.0)
                    dcorrx[0].SetBinError(ix,   0.001) # FIXME
                else:
                    f_qcd = (f_s[0] - f_l[0]*r_slp[0])/(1-r_slp[0])
                    df_s = 1/(1-r_slp[0])
                    df_l = r_slp[0]/(1-r_slp[0])
                    df_r = (f_qcd-f_l[0])/(1-r_slp[0])
                    df_qcd = sqrt((df_s*f_s[1])**2 + (df_l*f_l[1])**2 + (df_r*hypot(r_slp[1],r_slp[2]))**2)
                    dcorrx[0].SetBinContent(ix, f_qcd)
                    dcorrx[0].SetBinError(ix,   df_qcd) # FIXME
                if "TH1" in datay[0].ClassName():
                    for l,h in ('data',datay),('qcd',sigy),('ewk',bkgy):
                        text = "    Fake rate vs %s for %s in %s\n" % (p.name,l,xlabel)
                        text += "%3s   %8s  %8s    %6s +/- %6s\n" % ("bin", "xmin ", "xmax ", "value", "error ")
                        text += "%3s   %8s  %8s    %6s-----%6s\n" % ("---", "------", "------", "------", "------")
                        for b in xrange(1,h[0].GetNbinsX()+1):
                            n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                            f = n/float(d) if d > 0 else 0; 
                            if l == "data":
                                df = sqrt(f*(1-f)/d) if d > 0 else 0
                            else:
                                # get average weight of events (at numerator)
                                wavg = (h[0].GetBinError(b)**2) /h[0].GetBinContent(b) if h[0].GetBinError(b) else 1
                                df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                            text += "%3d   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (b, h[0].GetXaxis().GetBinLowEdge(b),h[0].GetXaxis().GetBinUpEdge(b), f,df)
                            h[0].SetBinContent(b, f) 
                            h[0].SetBinError(b, df)
                        c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 600, 400)
                        h[0].GetYaxis().SetTitle("Fake rate");
                        h[0].SetLineColor(color[l])
                        h[0].SetMarkerColor(color[l])
                        if l != "ewk": h[0].GetYaxis().SetRangeUser(0.0,0.4 if self._options.maxRatioRange[1] > 1 else self._options.maxRatioRange[1]);
                        else:          h[0].GetYaxis().SetRangeUser(0.8,1.0);
                        h[0].SetLineWidth(2)
                        h[0].Draw("E1")
                        h[0].GetXaxis().SetTitleOffset(0.9)
                        ROOT.gStyle.SetErrorX(0.5)
                        #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                        #for ext in self._options.printPlots.split(","):
                        #    if ext == "txt": 
                        #        dump = open("%s/FR_%s_%s_slice_%s.%s" % (self._denDir, p.name, l, xlabel, ext), "w")
                        #        dump.write(text)
                        #    else:
                        #        c1.Print("%s/FR_%s_%s_slice_%s.%s" % (self._denDir, p.name, l, xlabel, ext))
                    c1 = ROOT.TCanvas("FR_"+p.name+"_stack", p.name, 600, 400)
                    sigy[0].GetYaxis().SetRangeUser(0,1.0)
                    sigy[0].GetXaxis().SetTitleOffset(0.9)
                    sigy[0].Draw("E1")
                    bkgy[0].Draw("E1 SAME")
                    datay[0].Draw("E1 SAME")
                    lsub = ROOT.TLine(sigy[0].GetXaxis().GetXmin(), f_qcd, sigy[0].GetXaxis().GetXmax(), f_qcd);
                    lsub.SetLineWidth(3) 
                    lsub.SetLineColor(ROOT.kGreen+2) 
                    lsub.Draw("SAME")
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": continue
                        c1.Print("%s/FR_%s_%s_slice_%s.%s" % (self._denDir, p.name, "stack", xlabel, ext))
                else:
                    print "No idea how to handle a " + data[0].ClassName()
            for i in 0,1: sigx[i].SetLineColor(color['qcd'])
            for i in 0,1: bkgx[i].SetLineColor(color['ewk'])
            for i in 0,1: datax[i].SetLineColor(color['data'])
            for i in 0,1: dcorrx[i].SetLineColor(color['dsub'])
            for i in 0,1: sigx[i].SetMarkerColor(color['qcd'])
            for i in 0,1: bkgx[i].SetMarkerColor(color['ewk'])
            for i in 0,1: datax[i].SetMarkerColor(color['data'])
            for i in 0,1: dcorrx[i].SetMarkerColor(color['dsub'])
            for h in sigx+bkgx+datax+dcorrx:
                h.SetLineWidth(2)
                h.SetMarkerStyle(20)
                h.SetMarkerSize(1.0)
            c1 = ROOT.TCanvas("FR_"+p.name+"_stack", p.name, 600, 400)
            sigx[0].GetYaxis().SetRangeUser(0,0.4)
            sigx[0].GetYaxis().SetTitle("Fake rate");
            sigx[0].Draw("E1")
            bkgx[0].Draw("E1 SAME")
            datax[0].Draw("E1 SAME")
            dcorrx[0].Draw("E1 SAME")
            for ext in self._options.printPlots.split(","):
                if ext == "txt": continue
                c1.Print("%s/FR_%s_%s_final.%s" % (self._denDir, p.name, "stack", ext))
            if p.name == "l2d_met":
                c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 900, 800)
                c1.SetRightMargin(0.20)
                h2d = self._denFile.Get("pteta2d_data").Clone("h2d_template")
                for h1d,l in (sigx,"qcd"),(bkgx,"ewk"),(datax,"data"),(dcorrx,"dcorr"):
                    H1DToH2D(h1d[0],h2d,ROOT.fakeRateBin_Muons)
                    ROOT.gStyle.SetErrorX(0.5)
                    ROOT.gStyle.SetPaintTextFormat(".3f")
                    ROOT.gStyle.SetTextFont(62)
                    h2d.GetZaxis().SetTitle("Fake rate");
                    if l != "ewk": h2d.GetZaxis().SetRangeUser(0.0,0.4);
                    else:          h2d.GetZaxis().SetRangeUser(0.8,1.0);
                    h2d.Draw("COLZ TEXT90E")
                    h2d.SetMarkerSize(1.5)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": continue
                        c1.Print("%s/FR_%s_%s.%s" % (self._denDir, p.name, "unrolled_"+l, ext))
                    


class FakeRateMCSub1D:
    def __init__(self, plotFileName, denDir, numDir, myname, options):
        self._myname = myname
        self._plotFileName = plotFileName
        self._denDir = denDir
        self._numDir = numDir
        self._plots = PlotFile(plotFileName,options)
        self._numFile = ROOT.TFile.Open(self._numDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._denFile = ROOT.TFile.Open(self._denDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._options = options
    def scaleMC(self,proc,hnum,hden,mca):
        pass
    def systErrMC(self):
        return 0
    def makePlotsBySource(self,mca):
        for p in self._plots.plots():
            asig = mca.listSignals()[0]
            abkg = mca.listBackgrounds()[0]
            data = [self._numFile.Get(p.name + "_data"), self._denFile.Get(p.name + "_data")]
            if not data[0]: continue
            sig  = [self._numFile.Get(p.name + "_"+asig).Clone("snum"), self._denFile.Get(p.name + "_"+asig).Clone("sden")]
            bkg  = [self._numFile.Get(p.name + "_"+abkg).Clone("bnum"), self._denFile.Get(p.name + "_"+abkg).Clone("bden")]
            for i in 0,1:
                sig[i].Reset(); bkg[i].Reset();
            for proc in mca.listSignals():
                if self._numFile.Get(p.name + "_" + proc):
                    mnum = self._numFile.Get(p.name + "_" + proc)
                    mden = self._denFile.Get(p.name + "_" + proc)
                    self.scaleMC(proc,mnum,mden,mca)
                    sig[0].Add(mnum)
                    sig[1].Add(mden)
            for proc in mca.listBackgrounds():
                if self._numFile.Get(p.name + "_" + proc):
                    mnum = self._numFile.Get(p.name + "_" + proc)
                    mden = self._denFile.Get(p.name + "_" + proc)
                    self.scaleMC(proc,mnum,mden,mca)
                    bkg[0].Add(mnum)
                    bkg[1].Add(mden)
            mc = [ sig[0].Clone("mcnum"), sig[1].Clone("mcden") ]
            for i in 0,1 : mc[i].Add(bkg[i])
            dsub = [ data[0].Clone("subnum"), data[1].Clone("subden") ]
            if "TH1" in data[0].ClassName():
                color = { 'data':1, 'qcd':ROOT.kOrange+10, 'ewk':ROOT.kCyan+2, 'mc':ROOT.kOrange+4, 'dsub':ROOT.kGreen+3 }
                for l,h in ('data',data),('qcd',sig),('ewk',bkg),('mc',mc),('dsub',dsub):
                    text = "    Fake rate vs %s for %s\n" % (p.name,l)
                    text += "%3s   %8s  %8s    %6s +/- %6s\n" % ("bin", "xmin ", "xmax ", "value", "error ")
                    text += "%3s   %8s  %8s    %6s-----%6s\n" % ("---", "------", "------", "------", "------")
                    h.append(h[0].Clone(h[0].GetName()+"_FR_"))
                    for b in xrange(1,h[0].GetNbinsX()+1):
                        n,d = h[0].GetBinContent(b), h[1].GetBinContent(b)
                        if l == "dsub":
                            sn,sd = bkg[0].GetBinContent(b), bkg[1].GetBinContent(b)
                            f = (n-sn)/float(d-sd) if d-sd > 0 else 0; 
                            # now, to do the error propagation, take n = e_d * d, sn = e_p * dn
                            # with e_d and e_p distributed approximately as a binomial
                            # and add extra systematic s on sn,sd
                            e_d = n/float(d) if d > 0 else 0; 
                            d_e_d = sqrt(e_d*(1-e_d)/d) if d > 0 else 0 
                            e_p = sd/float(sd) if sd > 0 else 0
                            wavg = (bkg[0].GetBinError(b)**2) /bkg[0].GetBinError(b) if bkg[0].GetBinError(b) else 1
                            d_e_p = sqrt(e_p*(1-e_p)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and e_p > 0 and e_p <  1 else 0
                            d_s = self.systErrMC()
                            # derivatives
                            df_e_d = d/(d-sd)  if d-sd>0 else 0
                            df_e_p = sd/(d-sd) if d-sd>0 else 0
                            df_s   = abs(e_p-f)*sd/(d-sd) if d-sd>0 else 0
                            df = sqrt((d_e_d * df_e_d)**2 + (d_e_p * df_e_p)**2 + (d_s * df_s)**2)  
                        else:
                            f = n/float(d) if d > 0 else 0; 
                            if l == "data":
                                df = sqrt(f*(1-f)/d) if d > 0 else 0
                            else:
                                # get average weight of events (at numerator)
                                wavg = (h[0].GetBinError(b)**2) /h[0].GetBinError(b) if h[0].GetBinError(b) else 1
                                df = sqrt(f*(1-f)/(d/wavg)) if wavg > 0 and (d/wavg) > 0  and f > 0 and f <  1 else 0
                        text += "%3d   % 8.3f  % 8.3f    %.4f +/- %.4f\n" % (b, h[0].GetXaxis().GetBinLowEdge(b),h[0].GetXaxis().GetBinUpEdge(b), f,df)
                        h[2].SetBinContent(b, f) 
                        h[2].SetBinError(b, df)
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 600, 400)
                    h[2].GetYaxis().SetTitle("Fake rate");
                    h[2].SetLineColor(color[l])
                    h[2].SetMarkerColor(color[l])
                    if l != "ewk": h[2].GetYaxis().SetRangeUser(0.0,0.4 if self._options.maxRatioRange[1] > 1 else self._options.maxRatioRange[1]);
                    else:          h[2].GetYaxis().SetRangeUser(0.8,1.0);
                    h[2].SetLineWidth(2)
                    h[2].Draw("E1")
                    ROOT.gStyle.SetErrorX(0.5)
                    #doTinyCmsPrelim(hasExpo = False, textSize = 0.035)
                    for ext in self._options.printPlots.split(","):
                        if ext == "txt": 
                            dump = open("%s/FR_%s_%s_%s.%s" % (self._denDir, p.name, l, self._myname, ext), "w")
                            dump.write(text)
                        else:
                            c1.Print("%s/FR_%s_%s_%s.%s" % (self._denDir, p.name, l, self._myname, ext))
                c1 = ROOT.TCanvas("FR_"+p.name+"_stack", p.name, 600, 400)
                sig[2].Draw("E1")
                mc[2].Draw("E1 SAME")
                bkg[2].Draw("E1 SAME")
                data[2].Draw("E1 SAME")
                dsub[2].Draw("E1 SAME")
                for ext in self._options.printPlots.split(","):
                    if ext == "txt": continue
                    c1.Print("%s/FR_%s_%s_%s.%s" % (self._denDir, p.name, "stack", self._myname, ext))
                if p.name == "l2d":
                    c1 = ROOT.TCanvas("FR_"+p.name+"_"+l, p.name, 900, 800)
                    c1.SetRightMargin(0.20)
                    h2d = self._denFile.Get("pteta2d_data").Clone("h2d_template")
                    for h1d,l in (sig[2],"qcd"),(bkg[2],"ewk"),(data[2],"data"),(dsub[2],"dsub"):
                        H1DToH2D(h1d,h2d,ROOT.fakeRateBin_Muons)
                        ROOT.gStyle.SetErrorX(0.5)
                        ROOT.gStyle.SetPaintTextFormat(".3f")
                        ROOT.gStyle.SetTextFont(62)
                        h2d.GetZaxis().SetTitle("Fake rate");
                        if l != "ewk": h2d.GetZaxis().SetRangeUser(0.0,0.4);
                        else:          h2d.GetZaxis().SetRangeUser(0.8,1.0);
                        h2d.Draw("COLZ TEXT90E")
                        h2d.SetMarkerSize(1.5)
                        for ext in self._options.printPlots.split(","):
                            if ext == "txt": continue
                            c1.Print("%s/FR_%s_%s_%s.%s" % (self._denDir, p.name, "unrolled_"+l, self._myname, ext))
            else:
                print "No idea how to handle a " + data[0].ClassName()

class FakeRateUCx1D(FakeRateMCSub1D):
    def __init__(self,plotFileName, denDir, numDir, invMetDir, options):
        FakeRateMCSub1D.__init__(self,plotFileName, denDir, numDir, "UCxSub", options)
        self._invMetDir = invMetDir
        self._invMetFile = ROOT.TFile.Open(self._invMetDir+"/"+basename(self._plotFileName.replace(".txt",".root")))
        self._init = False
    def integral(self,h,xmin,xmax):
        n = 0
        for b in xrange(1,h.GetNbinsX()+1):
            if (h.GetXaxis().GetBinCenter(b) > xmin and h.GetXaxis().GetBinCenter(b) < xmax):
                n  += h.GetBinContent(b)
        return n
    def init(self,mca):
        # get the mtw plot in data
        hdata = self._invMetFile.Get("mtw_data")
        ndata = self.integral(hdata,60.,100.)
        # get the mtw plot in mc
        nmc   = 0
        for p in mca.listBackgrounds():
            hmc = self._invMetFile.Get("mtw_"+p)
            if hmc:
                nmc += self.integral(hmc,60.,100.) 
        self._sf = ndata/nmc
        print "UCx EWK normalization method: sf = ",self._sf
        self._init = True
    def scaleMC(self,proc,hnum,hden,mca):
        if mca.isBackground(proc):
            if not self._init: 
                self.init(mca)
            hnum.Scale(self._sf)
            hden.Scale(self._sf)
    def systErrMC(self):
        return 0.5*abs(1.0-self._sf)    
    
if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mc.txt cuts.txt plots.txt")
    addPlotMakerOptions(parser)
    parser.add_option("-o", "--out", dest="out", default=None, help="Output file name.");
    (options, args) = parser.parse_args()
    ROOT.gROOT.ProcessLine(".x tdrstyle.cc")
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)
    mca  = MCAnalysis(args[0],options)
    if len(args) == 5 and args[4] == "MET1Bin":
        if options.out == None: options.out = "FR_MET1Bin_"+basename(args[1]).replace(".txt","")+".root"
        FR = FakeRateMET1Bin(args[1], args[2], args[3], options) 
        FR.makePlotsBySource(mca)
    elif len(args) == 5 and args[4] == "MET1D":
        if options.out == None: options.out = "FR_MET1D_"+basename(args[1]).replace(".txt","")+".root"
        FR = FakeRateMET1D(args[1], args[2], args[3], options) 
        FR.makePlotsBySource(mca)
    elif len(args) == 6 and args[4] == "UCx":
        if options.out == None: options.out = "FR_UCx_"+basename(args[1]).replace(".txt","")+".root"
        FR = FakeRateUCx1D(args[1], args[2], args[3], args[5], options) 
        FR.makePlotsBySource(mca)
    elif len(args) == 5 and args[4] == "MC":
        print "MC"
        if options.out == None: options.out = "FR_MC_"+basename(args[1]).replace(".txt","")+".root"
        FR = MCFakeRate(args[1], args[2], args[3], options) 
        FR.makePlotsBySource(mca)
    else:
        if options.out == None: options.out = "FR_"+basename(args[1]).replace(".txt","")+".root"
        FR = FakeRateSimple(args[1], args[2], args[3], options) 
        FR.makePlotsBySource(mca)

