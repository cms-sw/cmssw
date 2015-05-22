import ROOT 
from math import *

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

def doLegend(x1,y1,x2,y2,textSize=0.035):
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetShadowColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(textSize)
    return leg

def doCMSSpam(text="CMS Preliminary", x0 = 0.17, textSize=0.033):
    doSpam(text             ,                     x0, .955, .400, .995, align=12, textSize=textSize)
    doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=textSize)


def histToGraph(hist):
    n = hist.GetNbinsX()
    ret = ROOT.TGraphErrors()
    for b in xrange(1,n+1):
        if hist.GetBinContent(b) == 0 and hist.GetBinError(b) == 0: continue
        ret.Set(ret.GetN()+1)
        ret.SetPoint(ret.GetN()-1, hist.GetXaxis().GetBinCenter(b),    hist.GetBinContent(b))
        ret.SetPointError(ret.GetN()-1, 0.5*hist.GetXaxis().GetBinWidth(b), hist.GetBinError(b))
    ret.GetXaxis().SetTitle(hist.GetXaxis().GetTitle())
    ret.GetYaxis().SetTitle(hist.GetYaxis().GetTitle())
    hist.graph = ret # attach so it survives
    return ret

def fitTGraph(graph, order=1, nToys=1000, nPoints=1000):
    if "TH1" in graph.ClassName():
        graph = histToGraph(graph)
    n = graph.GetN()
    #w = ROOT.RooWorkspace("w")
    #w.factory("x[%g,%g]" % (graph.GetX()[0], graph.GetX()[n-1]))
    #w.factory("_weight_[0,%g]" % (max([graph.GetY()[i]))
    #obs = ROOT.RooArgSet(w.var("x"))
    #dataset = ROOT.RooDataSet("d","",obs)
    #fpr 
    xmin = graph.GetX()[0]-graph.GetErrorXlow(0);
    xmax = graph.GetX()[n-1]+graph.GetErrorXhigh(n-1)
    poly = ROOT.TF1("poly", "pol%d" % order, xmin, xmax)
    resultptr = graph.Fit(poly, "QN0S EX0")
    result = resultptr.Get()
    ## now comes the fun: doing the bars
    cov = result.GetCovarianceMatrix()
    xvars, xlist, xset = [], ROOT.RooArgList(), ROOT.RooArgSet()
    parvec = ROOT.TVectorD(order+1)
    for i in xrange(order+1):
        xv = ROOT.RooRealVar("p%d" % i, "", result.Parameter(i)-10*result.ParError(i),  result.Parameter(i)+10*result.ParError(i))
        xvars.append(xv)
        xlist.add(xv)
        xset.add(xv)
        parvec[i] = result.Parameter(i)
    #print parvec
    #cov.Print()
    points = [ (xmin + i*(xmax-xmin)/(nPoints-1), []) for i in xrange(nPoints) ]
    mvg = ROOT.RooMultiVarGaussian("mvg","", xlist, parvec, cov)
    #print "Defined multi-variate gaussian "; mvg.Print("")
    data = mvg.generate(xset, nToys)
    #print "Generated dataset, now making bands"
    for i in xrange(data.numEntries()):
        e = data.get(i)
        for k in xrange(order+1):
            poly.SetParameter(k, e.getRealValue("p%d" % k))
        for x,ys in points:
            ys.append(poly.Eval(x))
    for k in xrange(order+1):
        poly.SetParameter(k, result.Parameter(k))
    band68 = ROOT.TGraphAsymmErrors(nPoints) 
    band95 = ROOT.TGraphAsymmErrors(nPoints) 
    for i,(x,ys) in enumerate(points):
        ys.sort()
        ny = len(ys)
        y  = poly.Eval(x)
        yhi68 = ys[int(min(ny-1, round(ny*(0.5+(0.68/2)))))]
        ylo68 = ys[int(max( 0,   round(ny*(0.5-(0.68/2)))))]
        yhi95 = ys[int(min(ny-1, round(ny*(0.5+(0.95/2)))))]
        ylo95 = ys[int(max( 0,   round(ny*(0.5-(0.95/2)))))]
        mean = sum(ys)/ny
        rms   = sqrt(sum([(yi-mean)**2 for yi in ys])/(ny-1)) 
        #print "at %3d: x = %.4f: %g [%g,%g], [%g,%g], %g +/- %g" % (i,x,y,ylo68,yhi68,ylo95,yhi95, mean, rms)
        band68.SetPoint(i, x, y)
        band95.SetPoint(i, x, y)
        dx = 0.5*(xmax-xmin)/(nPoints-1)
        #band68.SetPointError(i, dx, dx, rms, rms)
        #band95.SetPointError(i, 0, 0, 2*rms, 2*rms)
        band68.SetPointError(i, 0, 0, (y-ylo68), (yhi68-y))
        band95.SetPointError(i, 0, 0, (y-ylo95), (yhi95-y))
    # plot the best fit and bands
    poly.SetLineColor(ROOT.kRed)
    poly.SetLineWidth(2)
    band68.SetFillColor(ROOT.kGreen)
    band95.SetFillColor(ROOT.kYellow)
    band68.SetLineColor(ROOT.kRed)
    #band95.SetLineWidth(2)
    band95.Draw("E3 SAME")
    band68.Draw("E3 SAME")
    poly.Draw("SAME")
    # attach to graph so they survive
    graph.band68 = band68
    graph.band95 = band95
    graph.bestfit = poly
