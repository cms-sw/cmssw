import ROOT


def rawEfficiency(tree,numcut,dencut,xvar,xbins):
    if ROOT.gROOT.FindObject("htemp") != None: ROOT.gROOT.FindObject("htemp").Delete()
    if xbins[0] == "[":
        from array import array
        xvals = eval(xbins)
        hist2d = ROOT.TH2D("htemp","htemp",len(xvals)-1,array('f',xvals),2,array('f',[-0.5,0.5,1.5]))
    else:
        xn,xmin,xmax=xbins.split(",")
        hist2d = ROOT.TH2D("htemp","htemp",int(xn),float(xmin),float(xmax),2,-0.5,1.5)
    tree.Draw("(%s):(%s)>>htemp" % (numcut,xvar), dencut, "goff")
    cpcalc = ROOT.TEfficiency.ClopperPearson
    ret = ROOT.TGraphAsymmErrors(hist2d.GetNbinsX())
    for b in xrange(1,ret.GetN()+1):
        x0 = hist2d.GetXaxis().GetBinCenter(b)
        xmin, xmax = hist2d.GetXaxis().GetBinLowEdge(b), hist2d.GetXaxis().GetBinUpEdge(b)
        passing = int(hist2d.GetBinContent(b,2))
        total   = int(hist2d.GetBinContent(b,1) + passing)
        y0 = passing/float(total) if total else -99
        ymax = cpcalc(total,passing,0.6827,True ) if total else -99
        ymin = cpcalc(total,passing,0.6827,False) if total else -99
        ret.SetPoint(b-1, x0, y0)
        ret.SetPointError(b-1, x0-xmin, xmax-x0, y0-ymin,ymax-y0)
        print "at bin %3d: x = %8.2f, eff = %8d/%8d = %.3f  [%.3f, %.3f]" % (b, x0, passing,total, y0, ymin, ymax)
    return ret

def makeRatio(numeff,deneff):
    if numeff.GetN() != deneff.GetN(): raise RuntimeError, "Mismatching graphs"
    xn = numeff.GetX()
    yn = numeff.GetY()
    xd = deneff.GetX()
    yd = deneff.GetY()
    ratio = ROOT.TGraphAsymmErrors(numeff.GetN())
    unity = ROOT.TGraphAsymmErrors(numeff.GetN())
    for i in xrange(numeff.GetN()):
        if abs(xn[i]-xd[i]) > 1e-4: raise RuntimeError, "Mismatching graphs"
        if yd[i] <= 0:
            unity.SetPoint(i, xn[i], -99)
            ratio.SetPoint(i, xn[i], -99)
            unity.SetPointError(i, numeff.GetErrorXlow(i), numeff.GetErrorXhigh(i), 0,0)
            ratio.SetPointError(i, deneff.GetErrorXlow(i), deneff.GetErrorXhigh(i), 0,0)
        else:
            ratio.SetPoint(i, xn[i], yn[i]/yd[i])
            unity.SetPoint(i, xn[i], 1.0)
            ratio.SetPointError(i, numeff.GetErrorXlow(i), numeff.GetErrorXhigh(i), numeff.GetErrorYlow(i)/yd[i], numeff.GetErrorYhigh(i)/yd[i])
            unity.SetPointError(i, deneff.GetErrorXlow(i), deneff.GetErrorXhigh(i), deneff.GetErrorYlow(i)/yd[i], deneff.GetErrorYhigh(i)/yd[i])
            print "at bin %3d: x = %8.2f, ratio = %.3f/%.3f = %.3f -%.3f/+%.3f " % (i+1, xn[i], yn[i], yd[i], yn[i]/yd[i], numeff.GetErrorYlow(i)/yd[i], numeff.GetErrorYhigh(i)/yd[i])
    return [ ratio, unity ]

def makeRatios(effs):
    if len(effs) != 2: raise RuntimeError, "Not supported yet"
    return makeRatio(effs[0],effs[1])

def makeFrame(options):
    if ROOT.gROOT.FindObject("frame") != None: ROOT.gROOT.FindObject("frame").Delete()
    xvars,xbins = options.xvar
    if xbins[0] == "[":
        from array import array
        xvals = eval(xbins)
        hist1d = ROOT.TH1D("frame","frame",len(xvals)-1,array('f',xvals))
    else:
        xn,xmin,xmax=xbins.split(",")
        hist1d = ROOT.TH1D("frame","frame",int(xn),float(xmin),float(xmax))
    styleCommon(hist1d)
    return hist1d

def styleCommon(graph,options):
    graph.GetYaxis().SetRangeUser(options.yrange[0], options.yrange[1])
    graph.GetYaxis().SetDecimals(True)
    if options.xtitle: graph.GetXaxis().SetTitle(options.xtitle)
    if options.ytitle: graph.GetYaxis().SetTitle(options.ytitle)

def styleAsData(graph,options):
    styleCommon(graph,options)
    graph.SetMarkerStyle(20)
    graph.SetLineWidth(2)
    graph.SetLineColor(ROOT.kBlack)
    graph.SetMarkerColor(ROOT.kBlack)

def styleAsRef(graph,options):
    styleCommon(graph,options)
    if type(options.refcol) == str:
        if options.refcol[0] == "k": options.refcol = "ROOT."+options.refcol
        options.refcol = eval(options.refcol)
    graph.SetMarkerStyle(20)
    graph.SetFillColor(options.refcol)
    graph.SetMarkerStyle(27)
    graph.SetMarkerColor(ROOT.TColor.GetColorDark(options.refcol))

def stackEfficiencies(base,ref,options):
    styleAsData(base,options)
    styleAsRef(ref,options)
    ref.Draw("AE2");
    base.Draw("P SAME");

def plotRatios(effs,ratios,options):
    for e in effs:
        e.GetXaxis().SetLabelOffset(999) ## send them away
        e.GetXaxis().SetTitleOffset(999) ## in outer space
        e.GetYaxis().SetLabelSize(0.05)
    ratio, unity = ratios
    styleAsData(ratio,options)
    styleAsRef(unity,options)
    for r in ratios:
        r.GetYaxis().SetRangeUser(options.rrange[0],options.rrange[1]);
        r.GetXaxis().SetTitleSize(0.14)
        r.GetYaxis().SetTitleSize(0.14)
        r.GetXaxis().SetLabelSize(0.11)
        r.GetYaxis().SetLabelSize(0.11)
        r.GetYaxis().SetNdivisions(505)
        r.GetYaxis().SetTitle("ratio")
        r.GetYaxis().SetTitleOffset(0.52);
    #line = ROOT.TLine(unity.GetXaxis().GetXmin(),1,unity.GetXaxis().GetXmax(),1)
    #line.SetLineWidth(2);
    #line.SetLineColor(58);
    #line.Draw("L")
    unity.Draw("AE2");
    ratio.Draw("PZ SAME");
       

def plotEffs(effs,options):
    c1 = ROOT.TCanvas("c1", "c1", 600, (750 if options.doRatio else 600))
    c1.Draw()
    p1, p2 = c1, None # high and low panes
    # set borders, if necessary create subpads
    if len(effs) > 1 and options.doRatio:
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
    if len(effs) == 2:
        stackEfficiencies(effs[0],effs[1],options)
    else:
        styleAsData(effs[0],options)
        effs[0].Draw("AP")
    if len(effs) > 1 and options.doRatio:
        ratios = makeRatios(effs)
        p2.cd()
        plotRatios(effs,ratios,options)
    c1.Print("eff.png")
    

def addTnPEfficiencyOptions(parser):
    parser.add_option("-t", "--tree",    dest="tree", default='rec/t', help="Pattern for tree name");
    parser.add_option("-n", "--num",     dest="num", type="string", default="1", help="numerator")
    parser.add_option("-d", "--den",     dest="den", type="string", default="1", help="denominator")
    parser.add_option("-x", "--x-var",   dest="xvar", type="string", default=("LepGood_pt","10,0,50"), nargs=2, help="X var and bin")
    parser.add_option("--xtitle",   dest="xtitle", type="string", default=None, help="X title")
    parser.add_option("--ytitle",   dest="ytitle", type="string", default=None, help="Y title")
    parser.add_option("--refcol",   dest="refcol", type="string", default="kOrange+1", help="color for reference")
    parser.add_option("--yrange", dest="yrange", type="float", nargs=2, default=(0,1.025));
    parser.add_option("--rrange", dest="rrange", type="float", nargs=2, default=(0.78,1.24));
    parser.add_option("--doRatio", dest="doRatio", action="store_true", default=False, help="Add a ratio plot at the bottom")
    parser.add_option("--pdir", "--print-dir", dest="printDir", type="string", default="plots", help="print out plots in this directory");


if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] tree reftree")
    addTnPEfficiencyOptions(parser)
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print "You must specify at least one tree to fit"
    if len(args) > 2:
        print "You must specify at most two trees to fit"
    files = map( ROOT.TFile.Open, args )
    trees = [ f.Get(options.tree) for f in files ]
    effs =  [ rawEfficiency(t,options.num,options.den,options.xvar[0],options.xvar[1]) for t in trees ]
    ROOT.gROOT.SetBatch(True)
    ROOT.gROOT.ProcessLine(".x ~/tdrstyle.cc")
    plotEffs(effs,options)



