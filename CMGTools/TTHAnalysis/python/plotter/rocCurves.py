#!/usr/bin/env python
#from mcPlots import *
from CMGTools.TTHAnalysis.plotter.mcPlots import *


def hist2ROC1d(hsig,hbg):
    bins = hsig.GetNbinsX()+2
    si = [ hsig.GetBinContent(i) for i in xrange(bins) ]
    bi = [  hbg.GetBinContent(i) for i in xrange(bins) ]
    if hsig.GetMean() > hbg.GetMean():
        si.reverse(); bi.reverse()
    sums,sumb = sum(si), sum(bi)
    if sums == 0 or sumb == 0: 
        return None
    for i in xrange(1,bins): 
        si[i] += si[i-1]
        bi[i] += bi[i-1]
    fullsi, fullbi = si[:], bi[:]
    si, bi = [], [];
    for i in xrange(1,bins):
        # skip negative weights
        if len(si) > 0 and (fullsi[i] < si[-1] or fullbi[i] < bi[-1]):
            continue
        # skip repetitions
        if fullsi[i] != fullsi[i-1] or fullbi[i] != fullbi[i-1]:
            si.append(fullsi[i])
            bi.append(fullbi[i])
    if len(si) == 2: # just one WP + dummy (100%,100%)
        si = [si[0]]; bi = [ bi[0] ]
    bins = len(si)
    ret = ROOT.TGraph(bins)
    for i in xrange(bins):
        ret.SetPoint(i, bi[i]/sumb, si[i]/sums)
    ret.dim=1
    return ret

def hist2ROC2d(hsig,hbg):
    binsx = hsig.GetNbinsX()
    binsy = hsig.GetNbinsY()
    si = [ [ hsig.GetBinContent(ix,iy) for ix in xrange(1,binsx+1) ] for iy in xrange(1,binsy+1) ]
    bi = [ [  hbg.GetBinContent(ix,iy) for ix in xrange(1,binsx+1) ] for iy in xrange(1,binsy+1) ]
    sumsi = [ sum(s) for s in si ]
    sums,sumb = sum([sum(s) for s in si]), sum([sum(b) for b in bi])
    if hsig.GetMean(1) > hbg.GetMean(1): 
        si.reverse(); bi.reverse()
    for ix in xrange(0,binsx):
        if hsig.GetMean(2) > hbg.GetMean(2): 
            si[ix].reverse()
            bi[ix].reverse()
        for iy in xrange(1,binsx): 
            si[ix][iy] += si[ix][iy-1]
            bi[ix][iy] += bi[ix][iy-1]
    for ix in xrange(1,binsx):
        for iy in xrange(1,binsx): 
            si[ix][iy] += si[ix-1][iy]
            bi[ix][iy] += bi[ix-1][iy]
    bins = binsx*binsy
    ret = ROOT.TGraph(bins)
    i = 0
    for ix in xrange(binsx): 
        for iy in xrange(binsx): 
            ret.SetPoint(i, bi[ix][iy]/sumb, si[ix][iy]/sums)
            i += 1
    ret.dim=2
    return ret

def makeROC(plotmap,mca,sname="signal",bname="background"):
    sig = plotmap[sname]
    bkg = plotmap[bname]
    if sig.ClassName() == "TH1F":
        ret = hist2ROC1d(sig,bkg)
        if not ret: return ret
    elif sig.ClassName() == "TH2F":
        ret = hist2ROC2d(sig,bkg)
        if not ret: return ret
    else: raise RuntimeError, "Can't make a ROC from a "+sig.ClassName()
    ret.GetXaxis().SetTitle("Eff Background")
    ret.GetYaxis().SetTitle("Eff Signal")
    return ret

def addROCMakerOptions(parser):
    addMCAnalysisOptions(parser)
    parser.add_option("--select-plot", "--sP", dest="plotselect", action="append", default=[], help="Select only these plots out of the full file")
    parser.add_option("--exclude-plot", "--xP", dest="plotexclude", action="append", default=[], help="Exclude these plots from the full file")

def doLegend(rocs,textSize=0.035):
        (x1,y1,x2,y2) = (.6, .30 + textSize*max(len(rocs)-3,0), .93, .18)
        leg = ROOT.TLegend(x1,y1,x2,y2)
        leg.SetFillColor(0)
        leg.SetShadowColor(0)
        leg.SetTextFont(42)
        leg.SetTextSize(textSize)
        for key,val in rocs:
            leg.AddEntry(val, key, val.style)
        leg.Draw()
        ## assign it to a global variable so it's not deleted
        global legend_;
        legend_ = leg 
        return leg

def stackRocks(outname,outfile,rocs,xtit,ytit,options):
    allrocs = ROOT.TMultiGraph("all","all")
    for title,roc in rocs:
        allrocs.Add(roc,roc.style)
        outfile.WriteTObject(roc)
    c1 = ROOT.TCanvas("roc_canvas","roc_canvas")
    c1.SetGridy(options.showGrid)
    c1.SetGridx(options.showGrid)
    c1.SetLogx(options.logx)
    if options.xrange and options.yrange:
        frame = ROOT.TH1F("frame","frame",1000,options.xrange[0], options.xrange[1])
        frame.GetXaxis().SetTitle(xtit)
        frame.GetYaxis().SetTitle(ytit)
        frame.GetYaxis().SetRangeUser(options.yrange[0], options.yrange[1])
        frame.GetYaxis().SetDecimals(True)
        frame.Draw();
        for title,roc in rocs:
            roc.Draw(roc.style+" SAME")
    else:
        allrocs.Draw("APL");
        allrocs.GetXaxis().SetTitle(xtit)
        allrocs.GetYaxis().SetTitle(ytit)
        allrocs.GetYaxis().SetDecimals(True)
        if options.xrange:
            allrocs.GetXaxis().SetRangeUser(options.xrange[0], options.xrange[1])
        if options.yrange:
            allrocs.GetYaxis().SetRangeUser(options.yrange[0], options.yrange[1])
    leg = doLegend(rocs)
    if options.fontsize: leg.SetTextSize(options.fontsize)
    c1.Print(outname.replace(".root","")+".png")
    outfile.WriteTObject(c1,"roc_canvas")

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mc.txt cuts.txt plotfile.txt")
    addROCMakerOptions(parser)
    parser.add_option("-o", "--out", dest="out", default=None, help="Output file name. by default equal to plots -'.txt' +'.root'");
    parser.add_option("--xrange", dest="xrange", default=None, nargs=2, type='float', help="X axis range");
    parser.add_option("--yrange", dest="yrange", default=None, nargs=2, type='float', help="X axis range");
    parser.add_option("--xtitle", dest="xtitle", default="Eff Background", type='string', help="X axis title");
    parser.add_option("--ytitle", dest="ytitle", default="Eff Signal", type='string', help="Y axis title");
    parser.add_option("--fontsize", dest="fontsize", default=0, type='float', help="Legend font size");
    parser.add_option("--splitSig", dest="splitSig", default=False, action="store_true", help="Make one ROC per signal")
    parser.add_option("--splitBkg", dest="splitBkg", default=False, action="store_true", help="Make one ROC per background")
    parser.add_option("--grid", dest="showGrid", action="store_true", default=False, help="Show grid lines")
    parser.add_option("--logx", dest="logx", action="store_true", default=False, help="Log x-axis")
    parser.add_option("--groupBy",  dest="groupBy",  default="process",  type="string", help="Group by: variable, process")
    (options, args) = parser.parse_args()
    options.globalRebin = 1
    options.allowNegative = True # with the fine bins used in ROCs, one otherwise gets nonsensical results
    mca  = MCAnalysis(args[0],options)
    signals = mca.listSignals() if options.splitSig else [ "signal" ]
    backgrounds = mca.listBackgrounds() if options.splitBkg else [ "background" ]
    cut = CutsFile(args[1],options).allCuts()
    plots = PlotFile(args[2],options).plots()
    outname  = options.out if options.out else (args[2].replace(".txt","")+".root")
    outfile  = ROOT.TFile(outname,"RECREATE")
    ROOT.gROOT.ProcessLine(".x tdrstyle.cc")
    ROOT.gStyle.SetOptStat(0)
    pmaps = [  mca.getPlots(p,cut,makeSummary=True) for p in plots ]
    if len(signals+backgrounds)>2 and "variable" in options.groupBy:
        for ip,plot in enumerate(plots):
            pmap = pmaps[ip]
            rocs = []
            myname = outname.replace(".root","_%s.root" % plot.name)
            for i,(sig,bkg) in enumerate([(s,b) for s in signals for b in backgrounds ]):
                mytitle = ""; ptitle = None
                if len(signals)>1 and len(backgrounds)==1:
                    mytitle = mca.getProcessOption(sig,"Label",sig); ptitle = sig
                elif len(signals) == 1 and len(backgrounds)>1: 
                    mytitle = mca.getProcessOption(bkg,"Label",bkg); ptitle = bkg
                else:
                    mytitle = "%s/%s" % (mca.getProcessOption(sig,"Label",sig),mca.getProcessOption(bkg,"Label",bkg)) 
                roc = makeROC(pmap,mca,sname=sig,bname=bkg)
                if not roc: continue
                color = mca.getProcessOption(ptitle,"FillColor",SAFE_COLOR_LIST[i]) if ptitle else SAFE_COLOR_LIST[i]
                if roc.GetN() > 1 and roc.dim == 1 and not plot.getOption("Discrete",False):
                    roc.SetLineColor(color)
                    roc.SetMarkerColor(color)
                    roc.SetLineWidth(2)
                    roc.SetMarkerStyle(0)
                    roc.style = "L"
                else:
                    roc.SetMarkerColor(color)
                    roc.SetMarkerStyle(mca.getProcessOption(ptitle,"MarkerStyle",20 if roc.dim == 1 else 7) if ptitle else (20 if roc.dim == 1 else 7))
                    roc.SetMarkerSize(mca.getProcessOption(ptitle,"MarkerSize",1.0) if ptitle else 1.0)
                    roc.style = "P"
                roc.SetName("%s_%s_%s" % (plot.name,s,b))
                rocs.append((mytitle,roc))
            if len(rocs) == 0: continue
            stackRocks(myname,outfile,rocs,options.xtitle,options.ytitle,options)
    if "process" in options.groupBy:
        for (sig,bkg) in [(s,b) for s in signals for b in backgrounds ]:
            rocs = []
            myname = outname; xtit = options.xtitle; ytit = options.ytitle
            if len(signals)>1:     
                myname = myname.replace(".root","_%s.root" % sig)
                stit = mca.getProcessOption(sig,"Label",sig)
                ytit = ytit % stit if "%" in ytit else "%s (%s)" % (ytit,stit)
            if len(backgrounds)>1: 
                myname = myname.replace(".root","_%s.root" % bkg)
                btit = mca.getProcessOption(bkg,"Label",bkg);
                xtit = xtit % btit if "%" in xtit else "%s (%s)" % (xtit,btit)
            for i,plot in enumerate(plots):
                pmap = pmaps[i]
                roc = makeROC(pmap,mca,sname=sig,bname=bkg)
                if not roc: continue
                if roc.GetN() > 1 and roc.dim == 1 and not plot.getOption("Discrete",False):
                    roc.SetLineColor(plot.getOption("LineColor",SAFE_COLOR_LIST[i]))
                    roc.SetMarkerColor(plot.getOption("LineColor",SAFE_COLOR_LIST[i]))
                    roc.SetLineWidth(2)
                    roc.SetMarkerStyle(0)
                    roc.style = "L"
                else:
                    #print roc.GetX()[0],roc.GetY()[0],plot.name
                    roc.SetMarkerColor(plot.getOption("MarkerColor",SAFE_COLOR_LIST[i]))
                    roc.SetMarkerStyle(plot.getOption("MarkerStyle",20 if roc.dim == 1 else 7))
                    roc.SetMarkerSize(plot.getOption("MarkerSize",1.0))
                    roc.style = "P"
                #for ipoint in xrange(roc.GetN()):
                #    print "%-20s %6d    %.5f %.5f" % (plot.name,ipoint,roc.GetX()[ipoint],roc.GetY()[ipoint])
                roc.SetName(plot.name)
                rocs.append((plot.getOption("Title",plot.name),roc))
            if len(rocs) == 0: continue
            stackRocks(myname,outfile,rocs,xtit,ytit,options)
    outfile.Close()


