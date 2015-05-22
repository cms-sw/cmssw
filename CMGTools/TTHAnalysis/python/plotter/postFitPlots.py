#!/usr/bin/env python
import ROOT
ROOT.gROOT.SetBatch(True)

from math import *
from os.path import dirname,basename
from CMGTools.TTHAnalysis.tools.plotDecorations import *
from CMGTools.TTHAnalysis.plotter.mcPlots import *

mergeMap = { 
    "ttH_hww" : "ttH",
    "ttH_hzz" : "ttH",
    "ttH_htt" : "ttH",
    "TTWW" : "RARE",
    "TBZ" : "RARE",
    "WWqq" : "RARE",
    "WWDPI" : "RARE",
    "VVV" : "RARE",
    "TTGStar" : "TTZ",
}

options = None
if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mcaplot.txt mcafit.txt plot mlfile channel")
    addPlotMakerOptions(parser)
    (options, args) = parser.parse_args()
    options.path = "/afs/cern.ch/work/g/gpetrucc/TREES_250513_HADD"
    options.lumi = 19.5
    mcap = MCAnalysis(args[0],options)
    mca  = MCAnalysis(args[1],options)
    basedir = dirname(args[2]);
    infile = ROOT.TFile(args[2]);
    var    = args[3];
    mlfile = ROOT.TFile(args[4]);
    channel = args[5];
    ROOT.gROOT.ProcessLine(".x /afs/cern.ch/user/g/gpetrucc/cpp/tdrstyle.cc(0)")
    ROOT.gROOT.ForceStyle(False)
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaperSize(20.,25.)
    for O,MLD in ("prefit","prefit"), ("postfit","fit_b"):
      mldir  = mlfile.GetDirectory("shapes_"+MLD);
      outfile = ROOT.TFile(basedir + "/"+O+"_" + basename(args[2]), "RECREATE")
      processes = mca.listBackgrounds() + mca.listSignals()
      hdata = infile.Get(var+"_data")
      stack = ROOT.THStack(var+"_stack","")
      plots = {'data':hdata}
      if options.poisson:
            pdata = getDataPoissonErrors(hdata, False, True)
            hdata.poissonGraph = pdata ## attach it so it doesn't get deleted
      if "/4l" not in args[2]: mergeMap["ZZ"] = "RARE"
      for p in processes:
        pout = mergeMap[p] if p in mergeMap else p
        h = infile.Get(var+"_"+pout)
        if not h: 
            print "Missing %s_%s for %s" % (var,pout, p)
            continue
        h = h.Clone(var+"_"+p)
        h.SetDirectory(0)
        hpf = mldir.Get("%s/%s" % (channel,p))
        if not hpf: 
            if h.Integral() > 0 and p not in mergeMap: raise RuntimeError, "Could not find post-fit shape for %s" % p
            continue
        for b in xrange(1, h.GetNbinsX()+1):
            bi = b if "/4l" not in args[2] else b+1
            h.SetBinContent(b, hpf.GetBinContent(bi))
            h.SetBinError(b, hpf.GetBinError(bi))
        #pout = "ttH" if "ttH_" in p else p;
        #if pout in plots:
        #    plots[pout].Add(h)
        #else:
        if pout in plots:
            plots[pout].Add(h)
        else: 
            plots[pout] = h
            h.SetName(var+"_"+pout)
            stack.Add(h)
      htot = hdata.Clone(var+"_total")
      htotpf = mldir.Get(channel+"/total")
      hbkg = hdata.Clone(var+"_total_background")
      hbkgpf = mldir.Get(channel+"/total_background")
      for b in xrange(1, h.GetNbinsX()+1):
          bi = b if "/4l" not in args[2] else b+1
          htot.SetBinContent(b, htotpf.GetBinContent(bi))
          htot.SetBinError(b, htotpf.GetBinError(bi))
          hbkg.SetBinContent(b, hbkgpf.GetBinContent(bi))
          hbkg.SetBinError(b, hbkgpf.GetBinError(bi))
      for h in plots.values() + [htot]:
         outfile.WriteTObject(h)
      doRatio = True
      htot.GetYaxis().SetRangeUser(0, 1.8*max(htot.GetMaximum(), hdata.GetMaximum()))
      if "/4l" in args[2]: 
        htot.GetYaxis().SetRangeUser(0, 1.3*max(htot.GetMaximum(), hdata.GetMaximum()))
      ## Prepare split screen
      c1 = ROOT.TCanvas("c1", "c1", 600, 750); c1.Draw()
      c1.SetWindowSize(600 + (600 - c1.GetWw()), (750 + (750 - c1.GetWh())));
      p1 = ROOT.TPad("pad1","pad1",0,0.29,1,0.99);
      p1.SetBottomMargin(0.03);
      p1.Draw();
      p2 = ROOT.TPad("pad2","pad2",0,0,1,0.31);
      p2.SetTopMargin(0);
      p2.SetBottomMargin(0.3);
      p2.SetFillStyle(0);
      p2.Draw();
      p1.cd();
      ## Draw absolute prediction in top frame
      htot.Draw("HIST")
      if (htot.GetXaxis().GetTitle() == "N(jet"):
        htot.GetXaxis().SetTitle("N(jet)")
      #htot.SetLabelOffset(9999.0);
      #htot.SetTitleOffset(9999.0);
      stack.Draw("HIST F SAME")
      if options.poisson:
        hdata.poissonGraph.Draw("PZ SAME")
      else:
        hdata.Draw("E SAME")
      htot.Draw("AXIS SAME")
      hSigOutline = plots["ttH"].Clone()
      hSigOutline.SetLineWidth(5)
      hSigOutline.SetLineStyle(7)
      hSigOutline.SetLineColor(205)
      hSigOutline.SetFillColor(0)
      hSigOutline.SetFillStyle(1)
      hSigOutline.Scale(5)
      hSigOutline.Draw("HIST SAME")
      #doLegend(plots,mcap,corner=('TL' if '2LSS' in var or "/4l" in args[2] else 'TR'),textSize=0.037,cutoff=0.0001)
      leg = doLegend(plots,mcap,corner=('TL' if '2LSS' in var or "/4l" in args[2] else 'TR'),textSize=0.045,cutoff=0.0001)
      leg.AddEntry(hSigOutline, "ttH x 5", "L")
      lspam = "CMS ttH" #, options.lspam
      if "2lss" in args[2] and "/em/" in args[2]:
            lspam += r", e^{#pm}#mu^{#pm} channel"
      if "2lss" in args[2] and "/ee/" in args[2]:
            lspam += r", e^{#pm}e^{#pm} channel"
      if "2lss" in args[2] and "/mumu/" in args[2]:
            lspam += r", #mu^{#pm}#mu^{#pm} channel"
      if "/3l" in args[2]:
            lspam += ", 3l channel"
      if "/4l" in args[2]:
            lspam += ", 4l channel"
      doTinyCmsPrelim(hasExpo = False,textSize=(0.045 if doRatio else 0.033), xoffs=-0.03,
                      textLeft = lspam, textRight = options.rspam, lumi = options.lumi)
      ## Draw relaive prediction in the bottom frame
      p2.cd() 
      rdata,rnorm,rnorm2,rline = doRatioHists(PlotSpec(var,var,"",{}),plots,htot, htot, maxRange=options.maxRatioRange, fitRatio=options.fitRatio)
      c1.cd()
      c1.Print("%s/%s_%s.png" % (basedir,O,var))
      c1.Print("%s/%s_%s.pdf" % (basedir,O,var))
      del c1
      outfile.Close()
      dump = open("%s/%s_%s.txt" % (basedir,O,var), "w")
      pyields = { "background":[0,0], "data":[plots["data"].Integral(), plots["data"].Integral()]}
      argset =  mlfile.Get("norm_"+MLD)
      #argset.Print("V")
      for p in processes:
        pout = mergeMap[p] if p in mergeMap else p
        #h = infile.Get(var+"_"+pout)
        #if not h: continue
        if pout not in pyields: pyields[pout] = [0,0]
        rvar = argset.find("%s/%s" % (channel,p))
        if not rvar: continue
        pyields[pout][0] += rvar.getVal()       
        pyields[pout][1] += rvar.getError() if pout == "ttH" else rvar.getError()**2
        if not mcap.isSignal(pout): 
            pyields["background"][0] += rvar.getVal()       
            pyields["background"][1] += rvar.getError()**2
      for p in pyields.iterkeys():
        if p != "ttH": pyields[p][1] = sqrt(pyields[p][1])
      maxlen = max([len(mcap.getProcessOption(p,'Label',p)) for p in mcap.listSignals(allProcs=True) + mcap.listBackgrounds(allProcs=True)]+[7])
      fmt    = "%%-%ds %%9.2f +/- %%9.2f\n" % (maxlen+1)
      for p in mcap.listSignals(allProcs=True) + mcap.listBackgrounds(allProcs=True) + ["background","data"]:
        if p not in pyields: continue
        if p in ["background","data"]: dump.write(("-"*(maxlen+45))+"\n");
        dump.write(fmt % (mcap.getProcessOption(p,'Label',p) if p not in ["background","data"] else p.upper(), pyields[p][0], pyields[p][1]))
      dump.close() 
