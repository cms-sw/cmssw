#!/usr/bin/env python
#from mcAnalysis import *
from CMGTools.TTHAnalysis.plotter.mcAnalysis import *
import itertools

SAFE_COLOR_LIST=[
ROOT.kBlack, ROOT.kRed, ROOT.kGreen+2, ROOT.kBlue, ROOT.kMagenta+1, ROOT.kOrange+7, ROOT.kCyan+1, ROOT.kGray+2, ROOT.kViolet+5, ROOT.kSpring+5, ROOT.kAzure+1, ROOT.kPink+7, ROOT.kOrange+3, ROOT.kBlue+3, ROOT.kMagenta+3, ROOT.kRed+2,
]
def _unTLatex(string):
    return string.replace("#chi","x").replace("#mu","m")
class PlotFile:
    def __init__(self,fileName,options):
        self._options = options
        self._plots = []
        infile = open(fileName,'r')
        for line in infile:
            if re.match("\s*#.*", line) or len(line.strip())==0: continue
            while line.strip()[-1] == "\\":
                line = line.strip()[:-1] + infile.next()
            extra = {}
            if ";" in line:
                (line,more) = line.split(";")[:2]
                more = more.replace("\\,",";")
                for setting in [f.strip().replace(";",",") for f in more.split(',')]:
                    if "=" in setting: 
                        (key,val) = [f.strip() for f in setting.split("=")]
                        extra[key] = eval(val)
                    else: extra[setting] = True
            line = re.sub("#.*","",line) 
            field = [f.strip().replace(";",":") for f in line.replace("::",";;").replace("\\:",";").split(':')]
            if len(field) <= 2: continue
            if len(options.plotselect):
                skipMe = True
                for p0 in options.plotselect:
                    for p in p0.split(","):
                        if re.match(p+"$", field[0]): skipMe = False
                if skipMe: continue
            if len(options.plotexclude):
                skipMe = False
                for p0 in options.plotexclude:
                    for p in p0.split(","):
                        if re.match(p+"$", field[0]): skipMe = True
                if skipMe: continue
            if options.globalRebin: extra['rebinFactor'] = options.globalRebin
            self._plots.append(PlotSpec(field[0],field[1],field[2],extra))
    def plots(self):
        return self._plots[:]

def getDataPoissonErrors(h, drawZeroBins=False, drawXbars=False):
    xaxis = h.GetXaxis()
    q=(1-0.6827)/2.;
    points = []
    errors = []
    for i in xrange(h.GetNbinsX()):
        N = h.GetBinContent(i+1);
        if drawZeroBins or N > 0:
            x = xaxis.GetBinCenter(i+1);
            points.append( (x,N) )
            EYlow  = (N-ROOT.ROOT.Math.chisquared_quantile_c(1-q,2*N)/2.) if N > 0 else 0
            EYhigh = ROOT.ROOT.Math.chisquared_quantile_c(q,2*(N+1))/2.-N;
            EXhigh, EXlow = (xaxis.GetBinUpEdge(i+1)-x, x-xaxis.GetBinLowEdge(i+1)) if drawXbars else (0,0)
            errors.append( (EXlow,EXhigh,EYlow,EYhigh) )
    ret = ROOT.TGraphAsymmErrors(len(points))
    ret.SetName(h.GetName()+"_graph")
    for i,((x,y),(EXlow,EXhigh,EYlow,EYhigh)) in enumerate(zip(points,errors)):
        ret.SetPoint(i, x, y)
        ret.SetPointError(i, EXlow,EXhigh,EYlow,EYhigh)
    ret.SetLineWidth(h.GetLineWidth())
    ret.SetLineColor(h.GetLineColor())
    ret.SetLineStyle(h.GetLineStyle())
    ret.SetMarkerSize(h.GetMarkerSize())
    ret.SetMarkerColor(h.GetMarkerColor())
    ret.SetMarkerStyle(h.GetMarkerStyle())
    return ret

def doSpam(text,x1,y1,x2,y2,align=12,fill=False,textSize=0.033,_noDelete={}):
    cmsprel = ROOT.TPaveText(x1,y1,x2,y2,"NDC");
    cmsprel.SetTextSize(textSize);
    cmsprel.SetFillColor(0);
    cmsprel.SetFillStyle(1001 if fill else 0);
    cmsprel.SetLineStyle(2);
    cmsprel.SetLineColor(0);
    cmsprel.SetTextAlign(align);
    cmsprel.SetTextFont(42);
    cmsprel.AddText(text);
    cmsprel.Draw("same");
    _noDelete[text] = cmsprel; ## so it doesn't get deleted by PyROOT
    return cmsprel

def doTinyCmsPrelim(textLeft="_default_",textRight="_default_",hasExpo=False,textSize=0.033,lumi=None, xoffs=0):
    global options
    if textLeft  == "_default_": textLeft  = options.lspam
    if textRight == "_default_": textRight = options.rspam
    if lumi      == None       : lumi      = options.lumi
    if textLeft not in ['', None]:
        doSpam(textLeft, (.28 if hasExpo else .17)+xoffs, .955, .60+xoffs, .995, align=12, textSize=textSize)
    if textRight not in ['', None]:
        if "%(lumi)" in textRight: 
            textRight = textRight % { 'lumi':lumi }
        doSpam(textRight,.68+xoffs, .955, .99+xoffs, .995, align=32, textSize=textSize)

def reMax(hist,hist2,islog,factorLin=1.3,factorLog=2.0):
    if  hist.ClassName() == 'THStack':
        hist = hist.GetHistogram()
    max0 = hist.GetMaximum()
    max2 = hist2.GetMaximum()*(factorLog if islog else factorLin)
    if hasattr(hist2,'poissonGraph'):
       for i in xrange(hist2.poissonGraph.GetN()):
          max2 = max(max2, (hist2.poissonGraph.GetY()[i] + hist2.poissonGraph.GetErrorYhigh(i))*(factorLog if islog else factorLin))
    elif "TH1" in hist2.ClassName():
       for b in xrange(1,hist2.GetNbinsX()+1):
          max2 = max(max2, (hist2.GetBinContent(b) + hist2.GetBinError(b))*(factorLog if islog else factorLin))
    if max2 > max0:
        max0 = max2;
        if islog: hist.GetYaxis().SetRangeUser(0.9,max0)
        else:     hist.GetYaxis().SetRangeUser(0,max0)

def doDataNorm(pspec,pmap):
    if "data" not in pmap: return None
    total = sum([v.Integral() for k,v in pmap.iteritems() if k != 'data' and not hasattr(v,'summary')])
    sig = pmap["data"].Clone(pspec.name+"_data_norm")
    sig.SetFillStyle(0)
    sig.SetLineColor(1)
    sig.SetLineWidth(3)
    sig.SetLineStyle(2)
    if sig.Integral() > 0:
        sig.Scale(total/sig.Integral())
    sig.Draw("HIST SAME")
    return sig

def doStackSignalNorm(pspec,pmap,individuals,extrascale=1.0,norm=True):
    total = sum([v.Integral() for k,v in pmap.iteritems() if k != 'data' and not hasattr(v,'summary')])
    if options.noStackSig:
        total = sum([v.Integral() for k,v in pmap.iteritems() if not hasattr(v,'summary') and mca.isBackground(k) ])
    if individuals:
        sigs = []
        for sig in [pmap[x] for x in mca.listSignals() if pmap.has_key(x) and pmap[x].Integral() > 0]:
            sig = sig.Clone(sig.GetName()+"_norm")
            sig.SetFillStyle(0)
            sig.SetLineColor(sig.GetFillColor())
            sig.SetLineWidth(4)
            if norm: sig.Scale(total*extrascale/sig.Integral())
            sig.Draw("HIST SAME")
            sigs.append(sig)
        return sigs
    else:
        sig = None
        if "signal" in pmap: sig = pmap["signal"].Clone(pspec.name+"_signal_norm")
        else: 
            sigs = [pmap[x] for x in mca.listBackgrounds() if pmap.has_key(x) and pmap[x].Integral() > 0]
            sig = sigs[0].Clone(sigs.GetName()+"_norm")
        sig.SetFillStyle(0)
        sig.SetLineColor(206)
        sig.SetLineWidth(4)
        if norm and sig.Integral() > 0:
            sig.Scale(total*extrascale/sig.Integral())
        sig.Draw("HIST SAME")
        return [sig]

def doStackSigScaledNormData(pspec,pmap):
    if "data"       not in pmap: return (None,-1.0)
    if "signal"     not in pmap: return (None,-1.0)
    data = pmap["data"]
    sig = pmap["signal"].Clone("sig_refloat")
    bkg = None
    if "background" in pmap:
        bkg = pmap["background"]
    else:
        bkg = sig.Clone(); bkg.Reset()
    sf = (data.Integral()-bkg.Integral())/sig.Integral()
    sig.Scale(sf)
    sig.Add(bkg)
    sig.SetFillStyle(0)
    sig.SetLineColor(206)
    sig.SetLineWidth(3)
    sig.SetLineStyle(2)
    sig.Draw("HIST SAME")
    return (sig,sf)

def doScaleSigNormData(pspec,pmap,mca):
    if "data"       not in pmap: return -1.0
    if "signal"     not in pmap: return -1.0
    data = pmap["data"]
    sig = pmap["signal"].Clone("sig_refloat")
    bkg = None
    if "background" in pmap:
        bkg = pmap["background"]
    else:
        bkg = sig.Clone(); bkg.Reset()
    sf = (data.Integral()-bkg.Integral())/sig.Integral()
    signals = [ "signal" ] + mca.listSignals()
    for p,h in pmap.iteritems():
        if p in signals: h.Scale(sf)
    return sf

def doNormFit(pspec,pmap,mca):
    if "data" not in pmap: return -1.0
    data = pmap["data"]
    w = ROOT.RooWorkspace("w","w")
    x = w.factory("x[%g,%g]" % (data.GetXaxis().GetXmin(), data.GetXaxis().GetXmax()))
    x.setBins(data.GetNbinsX())
    obs = ROOT.RooArgList(w.var("x"))
    hdata = pmap['data']; hdata.killbins = False
    hmc = mergePlots('htemp', [v for (k,v) in pmap.iteritems() if k != 'data'])
    for b in xrange(1,hmc.GetNbinsX()+2):
        if hdata.GetBinContent(b) > 0 and hmc.GetBinContent(b) == 0:
            if not hdata.killbins:
                hdata = hdata.Clone()
                hdata.killbins = True
            for b2 in xrange(b-1,0,-1):
                if hmc.GetBinContent(b2) > 0:
                    hdata.SetBinContent(b2, hdata.GetBinContent(b2) + hdata.GetBinContent(b))
                    hdata.SetBinContent(b, 0)
                    break
            if hdata.GetBinContent(b) > 0:
                for b2 in xrange(b+1,hmc.GetNbinsX()+2):
                    if hmc.GetBinContent(b2) > 0:
                        hdata.SetBinContent(b2, hdata.GetBinContent(b2) + hdata.GetBinContent(b))
                        hdata.SetBinContent(b, 0)
                        break
            if hdata.GetBinContent(b) > 0: hdata.SetBinContent(b, 0)
    rdhs = {};
    w.imp = getattr(w, 'import')
    for p,h in pmap.iteritems():
        rdhs[p] = ROOT.RooDataHist("hist_"+p,"",obs,h if p != "data" else hdata)
        w.imp(rdhs[p])
    pdfs   = ROOT.RooArgList()
    coeffs = ROOT.RooArgList()
    constraints = ROOT.RooArgList()
    dontDelete = []
    procNormMap = {}
    for p in mca.listBackgrounds() + mca.listSignals():
        if p not in pmap: continue
        if pmap[p].Integral() == 0: continue
        hpdf = ROOT.RooHistPdf("pdf_"+p,"",ROOT.RooArgSet(x), rdhs[p])
        pdfs.add(hpdf); dontDelete.append(hpdf)
        if mca.getProcessOption(p,'FreeFloat',False):
            syst = mca.getProcessOption(p,'NormSystematic',0.0)
            normterm = w.factory('syst_%s[%g,%g,%g]' % (p, pmap[p].Integral(), 0.2*pmap[p].Integral(), 5*pmap[p].Integral() ))
            dontDelete.append((normterm,))
            coeffs.add(normterm)
            procNormMap[p] = normterm
        elif mca.getProcessOption(p,'NormSystematic',0.0) > 0:
            syst = mca.getProcessOption(p,'NormSystematic',0.0)
            normterm = w.factory('expr::norm_%s("%g*pow(%g,@0)",syst_%s[-5,5])' % (p, pmap[p].Integral(), 1+syst, p))
            constterm = w.factory('Gaussian::systpdf_%s(syst_%s,0,1)' % (p,p))
            dontDelete.append((normterm,constterm))
            coeffs.add(normterm)
            constraints.add(constterm)
            procNormMap[p] = normterm
        else:    
            normterm = w.factory('norm_%s[%g]' % (p, pmap[p].Integral()))
            dontDelete.append((normterm,))
            coeffs.add(normterm)
    pdfs.Print("V")
    coeffs.Print("V")
    addpdf = ROOT.RooAddPdf("tot","",pdfs,coeffs,False)
    model  = addpdf
    if constraints.getSize() > 0:
        constraints.add(addpdf)
        model = ROOT.RooProdPdf("prod","",constraints)
    result = model.fitTo( rdhs["data"], ROOT.RooFit.Save(1) )
    totsig, totbkg = None, None
    if "signal" in pmap and "signal" not in mca.listSignals(): 
        totsig = pmap["signal"]; totsig.Reset()
    if "background" in pmap and "background" not in mca.listBackgrounds(): 
        totbkg = pmap["background"]; totbkg.Reset()
    for p in mca.listBackgrounds() + mca.listSignals():
        if p in pmap and p in procNormMap:
           # setthe scale
           newscale = procNormMap[p].getVal()/pmap[p].Integral()
           pmap[p].Scale(newscale)
           # now get the 1 sigma
           nuis = w.var("syst_"+p);
           val,err = (nuis.getVal(), nuis.getError())
           v0 =  procNormMap[p].getVal()
           nuis.setVal(val+err)
           v1 =  procNormMap[p].getVal()
           nuis.setVal(val)
           #print [ p, val, err, v0, v1, (v1-v0)/v0, mca.getProcessOption(p,'NormSystematic',0.0) ]
           mca.setProcessOption(p,'NormSystematic', (v1-v0)/v0);
        # recompute totals
        if p in pmap:
            htot = totsig if mca.isSignal(p) else totbkg
            if htot != None:
                htot.Add(pmap[p])
                syst = mca.getProcessOption(p,'NormSystematic',0.0)
                if syst > 0:
                    for b in xrange(1,htot.GetNbinsX()+1):
                        htot.SetBinError(b, hypot(htot.GetBinError(b), pmap[p].GetBinContent(b)*syst))

def doRatioHists(pspec,pmap,total,totalSyst,maxRange,fitRatio=False):
    numkey = "data" 
    if "data" not in pmap: 
        if len(pmap) == 4 and 'signal' in pmap and 'background' in pmap:
            # do this first
            total.GetXaxis().SetLabelOffset(999) ## send them away
            total.GetXaxis().SetTitleOffset(999) ## in outer space
            total.GetYaxis().SetLabelSize(0.05)
            # then we can overwrite total with background
            numkey = 'signal'
            total     = pmap['background']
            totalSyst = pmap['background']
        else:    
            return (None,None,None,None)
    ratio = None
    if hasattr(pmap[numkey], 'poissonGraph'):
        ratio = pmap[numkey].poissonGraph.Clone("data_div"); 
        for i in xrange(ratio.GetN()):
            x    = ratio.GetX()[i]
            div  = total.GetBinContent(total.GetXaxis().FindBin(x))
            ratio.SetPoint(i, x, ratio.GetY()[i]/div if div > 0 else 0)
            ratio.SetPointError(i, ratio.GetErrorXlow(i), ratio.GetErrorXhigh(i), 
                                   ratio.GetErrorYlow(i)/div  if div > 0 else 0, 
                                   ratio.GetErrorYhigh(i)/div if div > 0 else 0) 
    else:
        ratio = pmap[numkey].Clone("data_div"); 
        ratio.Divide(total)
    unity  = totalSyst.Clone("sim_div");
    unity0 = total.Clone("sim_div");
    rmin, rmax =  1,1
    for b in xrange(1,unity.GetNbinsX()+1):
        e,e0,n = unity.GetBinError(b), unity0.GetBinError(b), unity.GetBinContent(b)
        unity.SetBinContent(b, 1 if n > 0 else 0)
        unity.SetBinError(b, e/n if n > 0 else 0)
        unity0.SetBinContent(b,  1 if n > 0 else 0)
        unity0.SetBinError(b, e0/n if n > 0 else 0)
        rmin = min([ rmin, 1-2*e/n if n > 0 else 1])
        rmax = max([ rmax, 1+2*e/n if n > 0 else 1])
    if ratio.ClassName() != "TGraphAsymmErrors":
        for b in xrange(1,unity.GetNbinsX()+1):
            if ratio.GetBinContent(b) == 0: continue
            rmin = min([ rmin, ratio.GetBinContent(b) - 2*ratio.GetBinError(b) ]) 
            rmax = max([ rmax, ratio.GetBinContent(b) + 2*ratio.GetBinError(b) ])  
    else:
        for i in xrange(ratio.GetN()):
            rmin = min([ rmin, ratio.GetY()[i] - 2*ratio.GetErrorYlow(i)  ]) 
            rmax = max([ rmax, ratio.GetY()[i] + 2*ratio.GetErrorYhigh(i) ])  
    if rmin < maxRange[0]: rmin = maxRange[0]; 
    if rmax > maxRange[1]: rmax = maxRange[1];
    if (rmax > 3 and rmax <= 3.4): rmax = 3.4
    if (rmax > 2 and rmax <= 2.4): rmax = 2.4
    unity.SetFillStyle(1001);
    unity.SetFillColor(ROOT.kCyan);
    unity.SetMarkerStyle(1);
    unity.SetMarkerColor(ROOT.kCyan);
    unity0.SetFillStyle(1001);
    unity0.SetFillColor(53);
    unity0.SetMarkerStyle(1);
    unity0.SetMarkerColor(53);
    ROOT.gStyle.SetErrorX(0.5);
    unity.Draw("E2");
    if fitRatio:
        from CMGTools.TTHAnalysis.tools.plotDecorations import fitTGraph
        fitTGraph(ratio,order=fitRatio)
        unity.SetFillStyle(3013);
        unity0.SetFillStyle(3013);
        unity.Draw("AXIS SAME");
        unity0.Draw("E2 SAME");
    else:
        if total != totalSyst: unity0.Draw("E2 SAME");
    unity.GetYaxis().SetRangeUser(rmin,rmax);
    unity.GetXaxis().SetTitleSize(0.14)
    unity.GetYaxis().SetTitleSize(0.14)
    unity.GetXaxis().SetLabelSize(0.11)
    unity.GetYaxis().SetLabelSize(0.11)
    unity.GetYaxis().SetNdivisions(505)
    unity.GetYaxis().SetDecimals(True)
    unity.GetYaxis().SetTitle("Data/Pred.")
    unity.GetYaxis().SetTitleOffset(0.52);
    total.GetXaxis().SetLabelOffset(999) ## send them away
    total.GetXaxis().SetTitleOffset(999) ## in outer space
    total.GetYaxis().SetLabelSize(0.05)
    #ratio.SetMarkerSize(0.7*ratio.GetMarkerSize()) # no it is confusing
    #$ROOT.gStyle.SetErrorX(0.0);
    line = ROOT.TLine(unity.GetXaxis().GetXmin(),1,unity.GetXaxis().GetXmax(),1)
    line.SetLineWidth(2);
    line.SetLineColor(58);
    line.Draw("L")
    ratio.Draw("E SAME" if ratio.ClassName() != "TGraphAsymmErrors" else "PZ SAME");
    return (ratio, unity, unity0, line)

def doStatTests(total,data,test,legendCorner):
    #print "Stat tests for %s:" % total.GetName()
    ksprob = data.KolmogorovTest(total,"XN")
    #print "\tKS  %.4f" % ksprob
    chi2l, chi2p, chi2gq, chi2lp, nb = 0, 0, 0, 0, 0
    for b in xrange(1,data.GetNbinsX()+1):
        oi = data.GetBinContent(b)
        ei = total.GetBinContent(b)
        dei = total.GetBinError(b)
        if ei <= 0: continue
        nb += 1
        chi2l += - 2*(oi*log(ei/oi)+(oi-ei) if oi > 0 else -ei)
        chi2p += (oi-ei)**2 / ei
        chi2gq += (oi-ei)**2 /(ei+dei**2)
        #chi2lp +=
    print "\tc2p  %.4f (%6.2f/%3d)" % (ROOT.TMath.Prob(chi2p,  nb), chi2p,  nb)
    print "\tc2l  %.4f (%6.2f/%3d)" % (ROOT.TMath.Prob(chi2l,  nb), chi2l,  nb)
    print "\tc2qg %.4f (%6.2f/%3d)" % (ROOT.TMath.Prob(chi2gq, nb), chi2gq, nb)
    #print "\tc2lp %.4f (%6.2f/%3d)" % (ROOT.TMath.Prob(chi2lp, nb), chi2lp, nb)
    chi2s = { "chi2l":chi2l, "chi2p":chi2p, "chi2gq":chi2gq, "chi2lp":chi2lp }
    if test in chi2s:
        chi2 = chi2s[test]
        pval = ROOT.TMath.Prob(chi2, nb)
        text = "#chi^{2} p-value %.3f" % pval if pval < 0.02 else "#chi^{2} p-value %.2f" % pval
    else:
        text = "Unknown test %s" % test
    if legendCorner == "TR":
        doSpam(text, .30, .85, .48, .93, align=32, textSize=0.05)
    elif legendCorner == "TL":
        doSpam(text, .75, .85, .93, .93, align=32, textSize=0.05)



legend_ = None;
def doLegend(pmap,mca,corner="TR",textSize=0.035,cutoff=1e-2,cutoffSignals=True,mcStyle="F",legWidth=0.18):
        if (corner == None): return
        total = sum([x.Integral() for x in pmap.itervalues()])
        sigEntries = []; bgEntries = []
        for p in mca.listSignals(allProcs=True):
            if mca.getProcessOption(p,'HideInLegend',False): continue
            if p in pmap and pmap[p].Integral() > (cutoff*total if cutoffSignals else 0): 
                lbl = mca.getProcessOption(p,'Label',p)
                sigEntries.append( (pmap[p],lbl,mcStyle) )
        backgrounds = mca.listBackgrounds(allProcs=True)
        for p in backgrounds:
            if mca.getProcessOption(p,'HideInLegend',False): continue
            if p in pmap and pmap[p].Integral() >= cutoff*total: 
                lbl = mca.getProcessOption(p,'Label',p)
                bgEntries.append( (pmap[p],lbl,mcStyle) )
        nentries = len(sigEntries) + len(bgEntries) + ('data' in pmap)

        (x1,y1,x2,y2) = (.93-legWidth, .75 - textSize*max(nentries-3,0), .93, .93)
        if corner == "TR":
            (x1,y1,x2,y2) = (.93-legWidth, .75 - textSize*max(nentries-3,0), .93, .93)
        elif corner == "BR":
            (x1,y1,x2,y2) = (.93-legWidth, .33 + textSize*max(nentries-3,0), .93, .15)
        elif corner == "TL":
            (x1,y1,x2,y2) = (.2, .75 - textSize*max(nentries-3,0), .2+legWidth, .93)
        
        leg = ROOT.TLegend(x1,y1,x2,y2)
        leg.SetFillColor(0)
        leg.SetShadowColor(0)
        leg.SetTextFont(42)
        leg.SetTextSize(textSize)
        if 'data' in pmap: 
            leg.AddEntry(pmap['data'], 'Data', 'LP')
        total = sum([x.Integral() for x in pmap.itervalues()])
        for (plot,label,style) in sigEntries: leg.AddEntry(plot,label,style)
        for (plot,label,style) in  bgEntries: leg.AddEntry(plot,label,style)
        leg.Draw()
        ## assign it to a global variable so it's not deleted
        global legend_;
        legend_ = leg 
        return leg

class PlotMaker:
    def __init__(self,tdir):
        self._options = options
        self._dir = tdir
        ROOT.gROOT.ProcessLine(".x tdrstyle.cc")
        ROOT.gROOT.ProcessLine(".L smearer.cc+")
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
    def run(self,mca,cuts,plots,makeStack=True,makeCanvas=True):
        sets = [ (None, 'all cuts', cuts.allCuts()) ]
        if not self._options.final:
            allcuts = cuts.sequentialCuts()
            if self._options.nMinusOne: allcuts = cuts.nMinusOneCuts()
            for i,(cn,cv) in enumerate(allcuts[:-1]): # skip the last one which is equal to all cuts
                cnsafe = "cut_%02d_%s" % (i, re.sub("[^a-zA-Z0-9_.]","",cn.replace(" ","_")))
                sets.append((cnsafe,cn,cv))
        for subname, title, cut in sets:
            print "cut set: ",title
            dir = self._dir
            if subname:
                if self._dir.Get(subname):
                    dir = self._dir.Get(subname)
                else:
                    dir = self._dir.mkdir(subname,title)
            dir.cd()
            for pspec in plots.plots():
                print "    plot: ",pspec.name
                pmap = mca.getPlots(pspec,cut,makeSummary=True)
                #
                # blinding policy
                blind = pspec.getOption('Blinded','None') if 'data' in pmap else 'None'
                if self._options.unblind == True: blind = 'None'
                xblind = [9e99,-9e99]
                if re.match(r'(bin|x)\s*([<>]?)\s*(\+|-)?\d+(\.\d+)?|(\+|-)?\d+(\.\d+)?\s*<\s*(bin|x)\s*<\s*(\+|-)?\d+(\.\d+)?', blind):
                    xfunc = (lambda h,b: b)             if 'bin' in blind else (lambda h,b : h.GetXaxis().GetBinCenter(b));
                    test  = eval("lambda bin : "+blind) if 'bin' in blind else eval("lambda x : "+blind) 
                    hdata = pmap['data']
                    for b in xrange(1,hdata.GetNbinsX()+1):
                        if test(xfunc(hdata,b)):
                            #print "blinding bin %d, x = [%s, %s]" % (b, hdata.GetXaxis().GetBinLowEdge(b), hdata.GetXaxis().GetBinUpEdge(b))
                            hdata.SetBinContent(b,0)
                            hdata.SetBinError(b,0)
                            xblind[0] = min(xblind[0],hdata.GetXaxis().GetBinLowEdge(b))
                            xblind[1] = max(xblind[1],hdata.GetXaxis().GetBinUpEdge(b))
                    #print "final blinded range x = [%s, %s]" % (xblind[0],xblind[1])
                elif blind != "None":
                    raise RuntimeError, "Unrecongnized value for 'Blinded' option, stopping here"
                #
                # Pseudo-data?
                if self._options.pseudoData:
                    if "data" in pmap: raise RuntimeError, "Can't use --pseudoData if there's also real data (maybe you want --xp data?)"
                    if self._options.pseudoData == "background":
                        pdata = pmap["background"]
                        pdata = pdata.Clone(str(pdata.GetName()).replace("_background","_data"))
                    elif self._options.pseudoData == "all":
                        pdata = pmap["background"]
                        pdata = pdata.Clone(str(pdata.GetName()).replace("_background","_data"))
                        if "signal" in pmap: pdata.Add(pmap["signal"])
                    else:
                        raise RuntimeError, "Pseudo-data option %s not supported" % self._options.pseudoData
                    if "TH1" in pdata.ClassName():
                        for i in xrange(1,pdata.GetNbinsX()+1):
                            pdata.SetBinContent(i, ROOT.gRandom.Poisson(pdata.GetBinContent(i)))
                            pdata.SetBinError(i, sqrt(pdata.GetBinContent(i)))
                    elif "TH2" in pdata.ClassName():
                        for ix in xrange(1,pdata.GetNbinsX()+1):
                          for iy in xrange(1,pdata.GetNbinsY()+1):
                            pdata.SetBinContent(ix, iy, ROOT.gRandom.Poisson(pdata.GetBinContent(ix, iy)))
                            pdata.SetBinError(ix, iy, sqrt(pdata.GetBinContent(ix, iy)))
                    else:
                        raise RuntimeError, "Can't make pseudo-data for %s" % pdata.ClassName()
                    pmap["data"] = pdata
                #
                if not makeStack: 
                    for k,v in pmap.iteritems():
                        if v.InheritsFrom("TH1"): v.SetDirectory(dir) 
                        dir.WriteTObject(v)
                    continue
                #
                stack = ROOT.THStack(pspec.name+"_stack",pspec.name)
                hists = [v for k,v in pmap.iteritems() if k != 'data']
                total = hists[0].Clone(pspec.name+"_total"); total.Reset()
                totalSyst = hists[0].Clone(pspec.name+"_totalSyst"); totalSyst.Reset()
                if self._options.plotmode == "norm": 
                    if 'data' in pmap:
                        total.GetYaxis().SetTitle(total.GetYaxis().GetTitle()+" (normalized)")
                    else:
                        total.GetYaxis().SetTitle("density/bin")
                    total.GetYaxis().SetDecimals(True)
                if options.scaleSignalToData: doScaleSigNormData(pspec,pmap,mca)
                elif options.fitData: doNormFit(pspec,pmap,mca)
                #
                for k,v in pmap.iteritems():
                    if v.InheritsFrom("TH1"): v.SetDirectory(dir) 
                    dir.WriteTObject(v)
                #
                for p in itertools.chain(reversed(mca.listBackgrounds(allProcs=True)), reversed(mca.listSignals(allProcs=True))):
                    if p in pmap: 
                        plot = pmap[p]
                        if plot.Integral() <= 0: continue
                        if mca.isSignal(p): plot.Scale(options.signalPlotScale)
                        if mca.isSignal(p) and options.noStackSig == True: continue 
                        if self._options.plotmode == "stack":
                            stack.Add(plot)
                            total.Add(plot)
                            totalSyst.Add(plot)
                            if mca.getProcessOption(p,'NormSystematic',0.0) > 0:
                                syst = mca.getProcessOption(p,'NormSystematic',0.0)
                                if "TH1" in plot.ClassName():
                                    for b in xrange(1,plot.GetNbinsX()+1):
                                        totalSyst.SetBinError(b, hypot(totalSyst.GetBinError(b), syst*plot.GetBinContent(b)))
                        else:
                            plot.SetLineColor(plot.GetFillColor())
                            plot.SetLineWidth(3)
                            plot.SetFillStyle(0)
                            if self._options.plotmode == "norm" and (plot.ClassName()[:2] == "TH"):
                                ref = pmap['data'].Integral() if 'data' in pmap else 1.0
                                plot.Scale(ref/plot.Integral())
                            stack.Add(plot)
                            total.SetMaximum(max(total.GetMaximum(),1.3*plot.GetMaximum()))
                        if self._options.errors and self._options.plotmode != "stack":
                            plot.SetMarkerColor(plot.GetFillColor())
                            plot.SetMarkerStyle(21)
                            plot.SetMarkerSize(1.5)
                        else:
                            plot.SetMarkerStyle(0)
                stack.Draw("GOFF")
                stack.GetYaxis().SetTitle(pspec.getOption('YTitle',"Events"))
                stack.GetXaxis().SetTitle(pspec.getOption('XTitle',pspec.name))
                stack.GetXaxis().SetNdivisions(pspec.getOption('XNDiv',510))
                dir.WriteTObject(stack)
                # 
                if not makeCanvas and not self._options.printPlots: continue
                doRatio = self._options.showRatio and ('data' in pmap or (self._options.plotmode != "stack" and len(pmap) == 4)) and ("TH2" not in total.ClassName())
                islog = pspec.hasOption('Logy'); 
                # define aspect ratio
                if doRatio: ROOT.gStyle.SetPaperSize(20.,25.)
                else:       ROOT.gStyle.SetPaperSize(20.,20.)
                # create canvas
                c1 = ROOT.TCanvas(pspec.name+"_canvas", pspec.name, 600, (750 if doRatio else 600))
                c1.Draw()
                p1, p2 = c1, None # high and low panes
                # set borders, if necessary create subpads
                if doRatio:
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
                else:
                    c1.SetWindowSize(600 + (600 - c1.GetWw()), 600 + (600 - c1.GetWh()));
                p1.SetLogy(islog)
                if pspec.hasOption('Logx'):
                    p1.SetLogx(True)
                    if p2: p2.SetLogx(True)
                    total.GetXaxis().SetNoExponent(True)
                    total.GetXaxis().SetMoreLogLabels(True)
                if islog: total.SetMaximum(2*total.GetMaximum())
                if not islog: total.SetMinimum(0)
                total.Draw("HIST")
                if self._options.plotmode == "stack":
                    stack.Draw("SAME HIST")
                    total.Draw("AXIS SAME")
                else: 
                    if self._options.errors:
                        ROOT.gStyle.SetErrorX(0.5)
                        stack.Draw("SAME E NOSTACK")
                    else:
                        stack.Draw("SAME HIST NOSTACK")
                if pspec.getOption('MoreY',1.0) > 1.0:
                    total.SetMaximum(pspec.getOption('MoreY',1.0)*total.GetMaximum())
                if 'data' in pmap: 
                    if options.poisson:
                        pdata = getDataPoissonErrors(pmap['data'], False, True)
                        pdata.Draw("PZ SAME")
                        pmap['data'].poissonGraph = pdata ## attach it so it doesn't get deleted
                    else:
                        pmap['data'].Draw("E SAME")
                    reMax(total,pmap['data'],islog)
                    if xblind[0] < xblind[1]:
                        blindbox = ROOT.TBox(xblind[0],total.GetYaxis().GetXmin(),xblind[1],total.GetMaximum())
                        blindbox.SetFillColor(ROOT.kBlue+3)
                        blindbox.SetFillStyle(3944)
                        blindbox.Draw()
                        xblind.append(blindbox) # so it doesn't get deleted
                    if options.doStatTests:
                        doStatTests(totalSyst, pmap['data'], options.doStatTests, legendCorner=pspec.getOption('Legend','TR'))
                if pspec.hasOption('YMin') and pspec.hasOption('YMax'):
                    total.GetYaxis().SetRangeUser(pspec.getOption('YMin',1.0), pspec.getOption('YMax',1.0))
                legendCutoff = pspec.getOption('LegendCutoff', 1e-5 if c1.GetLogy() else 1e-2)
                if self._options.plotmode == "norm": legendCutoff = 0 
                doLegend(pmap,mca,corner=pspec.getOption('Legend','TR'),
                                  cutoff=legendCutoff, mcStyle=("F" if self._options.plotmode == "stack" else "L"),
                                  cutoffSignals=not(options.showSigShape or options.showIndivSigShapes or options.showSFitShape), 
                                  textSize=(0.045 if doRatio else 0.035),
                                  legWidth=options.legendWidth)
                doTinyCmsPrelim(hasExpo = total.GetMaximum() > 9e4 and not c1.GetLogy(),textSize=(0.045 if doRatio else 0.033))
                signorm = None; datnorm = None; sfitnorm = None
                if options.showSigShape or options.showIndivSigShapes or options.showIndivSigs: 
                    signorms = doStackSignalNorm(pspec,pmap,options.showIndivSigShapes or options.showIndivSigs,extrascale=options.signalPlotScale, norm=not options.showIndivSigs)
                    for signorm in signorms:
                        signorm.SetDirectory(dir); dir.WriteTObject(signorm)
                        reMax(total,signorm,islog)
                if options.showDatShape: 
                    datnorm = doDataNorm(pspec,pmap)
                    if datnorm != None:
                        datnorm.SetDirectory(dir); dir.WriteTObject(datnorm)
                        reMax(total,datnorm,islog)
                if options.showSFitShape: 
                    (sfitnorm,sf) = doStackSigScaledNormData(pspec,pmap)
                    if sfitnorm != None:
                        sfitnorm.SetDirectory(dir); dir.WriteTObject(sfitnorm)
                        reMax(total,sfitnorm,islog)
                if options.flagDifferences and len(pmap) == 4:
                    new = pmap['signal']
                    ref = pmap['background']
                    if "TH1" in new.ClassName():
                        for b in xrange(1,new.GetNbinsX()+1):
                            if new.GetBinContent(b) != ref.GetBinContent(b):
                                print "Plot: difference found in %s, bin %d" % (pspec.name, b)
                                p1.SetFillColor(ROOT.kYellow-10)
                                if p2: p2.SetFillColor(ROOT.kYellow-10)
                                break
                if makeCanvas: dir.WriteTObject(c1)
                rdata,rnorm,rnorm2,rline = (None,None,None,None)
                if doRatio:
                    p2.cd(); 
                    rdata,rnorm,rnorm2,rline = doRatioHists(pspec,pmap,total,totalSyst, maxRange=options.maxRatioRange, fitRatio=options.fitRatio)
                if self._options.printPlots:
                    for ext in self._options.printPlots.split(","):
                        fdir = self._options.printDir;
                        if subname: fdir += "/"+subname;
                        if not os.path.exists(fdir): 
                            os.makedirs(fdir); 
                            if os.path.exists("/afs/cern.ch"): os.system("cp /afs/cern.ch/user/g/gpetrucc/php/index.php "+fdir)
                        if ext == "txt":
                            dump = open("%s/%s.%s" % (fdir, pspec.name, ext), "w")
                            maxlen = max([len(mca.getProcessOption(p,'Label',p)) for p in mca.listSignals(allProcs=True) + mca.listBackgrounds(allProcs=True)]+[7])
                            fmt    = "%%-%ds %%9.2f +/- %%9.2f (stat)" % (maxlen+1)
                            for p in mca.listSignals(allProcs=True) + mca.listBackgrounds(allProcs=True) + ["signal", "background"]:
                                if p not in pmap: continue
                                plot = pmap[p]
                                if plot.Integral() <= 0: continue
                                norm = plot.Integral()
                                if p not in ["signal","background"] and mca.isSignal(p): norm /= options.signalPlotScale # un-scale what was scaled
                                stat = sqrt(sum([plot.GetBinError(b)**2 for b in xrange(1,plot.GetNbinsX()+1)]))
                                syst = norm * mca.getProcessOption(p,'NormSystematic',0.0) if p not in ["signal", "background"] else 0;
                                if p == "signal": dump.write(("-"*(maxlen+45))+"\n");
                                dump.write(fmt % (_unTLatex(mca.getProcessOption(p,'Label',p) if p not in ["signal", "background"] else p.upper()), norm, stat))
                                if syst: dump.write(" +/- %9.2f (syst)"  % syst)
                                dump.write("\n")
                            if 'data' in pmap: 
                                dump.write(("-"*(maxlen+45))+"\n");
                                dump.write(("%%%ds %%7.0f\n" % (maxlen+1)) % ('DATA', pmap['data'].Integral()))
                            dump.close()
                        else:
                            if "TH2" in total.ClassName() or "TProfile2D" in total.ClassName():
                                for p in mca.listSignals(allProcs=True) + mca.listBackgrounds(allProcs=True) + ["signal", "background"]:
                                    if p not in pmap: continue
                                    plot = pmap[p]
                                    c1.SetRightMargin(0.20)
                                    plot.SetContour(100)
                                    plot.Draw("COLZ")
                                    c1.Print("%s/%s_%s.%s" % (fdir, pspec.name, p, ext))
                            else:
                                c1.Print("%s/%s.%s" % (fdir, pspec.name, ext))
                c1.Close()
def addPlotMakerOptions(parser):
    addMCAnalysisOptions(parser)
    parser.add_option("--ss",  "--scale-signal", dest="signalPlotScale", default=1.0, type="float", help="scale the signal in the plots by this amount");
    #parser.add_option("--lspam", dest="lspam",   type="string", default="CMS Simulation", help="Spam text on the right hand side");
    parser.add_option("--lspam", dest="lspam",   type="string", default="CMS Preliminary", help="Spam text on the right hand side");
    parser.add_option("--rspam", dest="rspam",   type="string", default="#sqrt{s} = 13 TeV, L = %(lumi).1f fb^{-1}", help="Spam text on the right hand side");
    parser.add_option("--print", dest="printPlots", type="string", default="png,pdf,txt", help="print out plots in this format or formats (e.g. 'png,pdf,txt')");
    parser.add_option("--pdir", "--print-dir", dest="printDir", type="string", default="plots", help="print out plots in this directory");
    parser.add_option("--showSigShape", dest="showSigShape", action="store_true", default=False, help="Superimpose a normalized signal shape")
    parser.add_option("--showIndivSigShapes", dest="showIndivSigShapes", action="store_true", default=False, help="Superimpose normalized shapes for each signal individually")
    parser.add_option("--showIndivSigs", dest="showIndivSigs", action="store_true", default=False, help="Superimpose shapes for each signal individually (normalized to their expected event yield)")
    parser.add_option("--noStackSig", dest="noStackSig", action="store_true", default=False, help="Don't add the signal shape to the stack (useful with --showSigShape)")
    parser.add_option("--showDatShape", dest="showDatShape", action="store_true", default=False, help="Stack a normalized data shape")
    parser.add_option("--showSFitShape", dest="showSFitShape", action="store_true", default=False, help="Stack a shape of background + scaled signal normalized to total data")
    parser.add_option("--showRatio", dest="showRatio", action="store_true", default=False, help="Add a data/sim ratio plot at the bottom")
    parser.add_option("--fitRatio", dest="fitRatio", type="int", default=False, help="Fit the ratio with a polynomial of the specified order")
    parser.add_option("--scaleSigToData", dest="scaleSignalToData", action="store_true", default=False, help="Scale all signal processes so that the overall event yield matches the observed one")
    parser.add_option("--fitData", dest="fitData", action="store_true", default=False, help="Perform a fit to the data")
    parser.add_option("--maxRatioRange", dest="maxRatioRange", type="float", nargs=2, default=(0.0, 5.0), help="Min and max for the ratio")
    parser.add_option("--doStatTests", dest="doStatTests", type="string", default=None, help="Do this stat test: chi2p (Pearson chi2), chi2l (binned likelihood equivalent of chi2)")
    parser.add_option("--plotmode", dest="plotmode", type="string", default="stack", help="Show as stacked plot (stack), a non-stacked comparison (nostack) and a non-stacked comparison of normalized shapes (norm)")
    parser.add_option("--rebin", dest="globalRebin", type="int", default="0", help="Rebin all plots by this factor")
    parser.add_option("--poisson", dest="poisson", action="store_true", default=False, help="Draw Poisson error bars")
    parser.add_option("--unblind", dest="unblind", action="store_true", default=False, help="Unblind plots irrespectively of plot file")
    parser.add_option("--select-plot", "--sP", dest="plotselect", action="append", default=[], help="Select only these plots out of the full file")
    parser.add_option("--exclude-plot", "--xP", dest="plotexclude", action="append", default=[], help="Exclude these plots from the full file")
    parser.add_option("--legendWidth", dest="legendWidth", type="float", default=0.25, help="Width of the legend")
    parser.add_option("--flagDifferences", dest="flagDifferences", action="store_true", default=False, help="Flag plots that are different (when using only two processes, and plotmode nostack")
    parser.add_option("--pseudoData", dest="pseudoData", type="string", default=None, help="If set to 'background' or 'all', it will plot also a pseudo-dataset made from background (or signal+background) with Poisson fluctuations in each bin.")

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mc.txt cuts.txt plots.txt")
    addPlotMakerOptions(parser)
    parser.add_option("-o", "--out", dest="out", default=None, help="Output file name. by default equal to plots -'.txt' +'.root'");
    (options, args) = parser.parse_args()
    mca  = MCAnalysis(args[0],options)
    cuts = CutsFile(args[1],options)
    plots = PlotFile(args[2],options)
    outname  = options.out if options.out else (args[2].replace(".txt","")+".root")
    if (not options.out) and options.printDir:
        outname = options.printDir + "/"+os.path.basename(args[2].replace(".txt","")+".root")
    if os.path.dirname(outname) and not os.path.exists(os.path.dirname(outname)):
        os.system("mkdir -p "+os.path.dirname(outname))
        if os.path.exists("/afs/cern.ch"): os.system("cp /afs/cern.ch/user/g/gpetrucc/php/index.php "+os.path.dirname(outname))
    print "Will save plots to ",outname
    fcut = open(re.sub("\.root$","",outname)+"_cuts.txt","w")
    fcut.write("%s\n" % cuts); fcut.close()
    os.system("cp %s %s " % (args[2], re.sub("\.root$","",outname)+"_plots.txt"))
    os.system("cp %s %s " % (args[0], re.sub("\.root$","",outname)+"_mca.txt"))
    #fcut = open(re.sub("\.root$","",outname)+"_cuts.txt")
    #fcut.write(cuts); fcut.write("\n"); fcut.close()
    outfile  = ROOT.TFile(outname,"RECREATE")
    plotter = PlotMaker(outfile)
    plotter.run(mca,cuts,plots)
    outfile.Close()


