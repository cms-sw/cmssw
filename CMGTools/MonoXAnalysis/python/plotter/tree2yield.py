#!/usr/bin/env python
from math import *
import re
import os, os.path
from array import array

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)

from copy import *

from CMGTools.TTHAnalysis.plotter.mcCorrections import *
from CMGTools.TTHAnalysis.plotter.fakeRate import *

if "/functions_cc.so" not in ROOT.gSystem.GetLibraries(): 
    ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/MonoXAnalysis/python/plotter/functions.cc+" % os.environ['CMSSW_BASE']);

def scalarToVector(x):
    x0 = x
    x = re.sub(r"(LepGood|Lep|JetFwd|JetClean|Jet|GenTop|SV)(\d)_(\w+)", lambda m : "%s_%s[%d]" % (m.group(1),m.group(3),int(m.group(2))-1), x)
    x = re.sub(r"\bmet\b", "met_pt", x)
    return x

class CutsFile:
    def __init__(self,txtfileOrCuts,options=None):
        if type(txtfileOrCuts) == list:
            self._cuts = deepcopy(txtfileOrCuts[:])
        elif isinstance(txtfileOrCuts,CutsFile):
            self._cuts = deepcopy(txtfileOrCuts.cuts())
        else:
            self._cuts = []
            file = open(txtfileOrCuts, "r")
            if not file: raise RuntimeError, "Cannot open "+txtfileOrCuts+"\n"
            for cr,cn,cv in options.cutsToAdd:
                if re.match(cr,"entry point"): self._cuts.append((cn,cv))
            for line in file:
              try:
                if len(line.strip()) == 0 or line.strip()[0] == '#': continue
                while line.strip()[-1] == "\\":
                    line = line.strip()[:-1] + file.next()
                (name,cut) = [x.strip().replace(";",":") for x in line.replace("\:",";").split(":")]
                if name == "entry point" and cut == "1": continue
                if options.startCut and not re.search(options.startCut,name): continue
                if options.startCut and re.search(options.startCut,name): options.startCut = None
                self._cuts.append((name,cut))
                for cr,cn,cv in options.cutsToAdd:
                    if re.match(cr,name): self._cuts.append((cn,cv))
                if options.upToCut and re.search(options.upToCut,name):
                    break
              except ValueError, e:
                print "Error parsing cut line [%s]" % line.strip()
                raise 
            for ci in options.cutsToInvert:  self.invert(ci)
            for ci in options.cutsToExclude: self.remove(ci)
            for cr,cn,cv in options.cutsToReplace: self.replace(cr,cn,cv)
    def __str__(self):
        newstring = ""
        for cut in self._cuts:
            newstring += "{0} : {1}\n".format(cut[0],cut[1])
        return newstring[:-1]
    def remove(self,cut):
        self._cuts = [(cn,cv) for (cn,cv) in self._cuts if not re.search(cut,cn)]
        return self
    def invert(self,cut):
        for i,(cn,cv) in enumerate(self._cuts[:]):
            if re.search(cut,cn):
                if cn.startswith("not ") and re.match(r"!\(.*\)", cv):
                    self._cuts[i] = (cn[4:], cv[2:-1])
                else:
                    self._cuts[i] = ("not "+cn, "!("+cv+")")
        return self
    def replace(self,cut,newname,newcut):       
        for i,(cn,cv) in enumerate(self._cuts[:]):
            if re.search(cut,cn):
                self._cuts[i] = (newname, newcut)
        return self
    def cuts(self):
        return self._cuts[:]
    def sequentialCuts(self):
        if len(self._cuts) == 0: return []
        ret = [ (self._cuts[0][0], "(%s)" % self._cuts[0][1]) ]
        for (cn,cv) in self._cuts[1:]:
            ret.append( ( cn, "%s && (%s)" % (ret[-1][1], cv) ) )
        return ret
    def nMinusOne(self):
        return CutsFile(self.nMinusOneCuts())
    def nMinusOneCuts(self):
        ret = []
        for cn,cv in self._cuts[1:]:
            nm1 = " && ".join("(%s)" % cv1 for cn1,cv1 in self._cuts if cn1 != cn)
            ret.append(("all but "+cn, nm1))
        return ret
    def allCuts(self,n=-1):
        return " && ".join("(%s)" % x[1] for x in (self._cuts[0:n+1] if n != -1 and n+1 < len(self._cuts) else self._cuts))
    def addAfter(self,cut,newname,newcut):
        for i,(cn,cv) in enumerate(self._cuts[:]):
            if re.search(cut,cn):
                self._cuts.insert(i+1,(newname, newcut))
                break
        return self
    def insert(self,index,newname,newcut):
        self._cuts.insert(index,(newname, newcut))
        return self
    def add(self,newname,newcut):
        self._cuts.append((newname,newcut))
        return self
    def setParams(self,paramMap):
        self._cuts = [ (cn.format(**paramMap), cv.format(**paramMap)) for (cn,cv) in self._cuts ]
    def cartesianProduct(self,other):
        return CutsFile( [ ("%s && %s" % (cn1,cn2), "(%s) && (%s)" % (cv1,cv2)) for (cn1,cv1) in self._cuts for (cn2,cv2) in other.cuts() ] )

class PlotSpec:
    def __init__(self,name,expr,bins,opts):
        self.name = name
        self.expr = expr
        self.bins = bins
        self.opts = opts
    def hasOption(self,name):
        return (name in self.opts)
    def getOption(self,name,default=None):
        return self.opts[name] if (name in self.opts) else default

class TreeToYield:
    def __init__(self,root,options,scaleFactor=1.0,name=None,cname=None,settings={},treename=None):
        self._name  = name  if name != None else root
        self._cname = cname if cname != None else self._name
        self._fname = root
        self._isInit = False
        self._options = options
        self._treename = treename if treename else options.tree
        self._weight  = (options.weight and 'data' not in self._name and '2012' not in self._name and '2011' not in self._name )
        self._isdata = 'data' in self._name
        self._weightString  = options.weightString if not self._isdata else "1"
        self._scaleFactor = scaleFactor
        self._fullYield = 0 # yield of the full sample, as if it passed the full skim and all cuts
        self._settings = settings
        loadMCCorrections(options)            ## make sure this is loaded
        self._mcCorrs = globalMCCorrections() ##  get defaults
        if 'SkipDefaultMCCorrections' in settings: ## unless requested to 
            self._mcCorrs = []                     ##  skip them
        if self._isdata: 
            self._mcCorrs = [c for c in self._mcCorrs if c.alsoData] ## most don't apply to data, some do 
        if 'MCCorrections' in settings:
            self._mcCorrs = self._mcCorrs[:] # make copy
            for cfile in settings['MCCorrections'].split(','): 
                self._mcCorrs.append( MCCorrections(cfile) )
        if 'FakeRate' in settings:
            self._FR = FakeRate(settings['FakeRate'])
            ## add additional weight correction 
            self._weightString += "* (" + self.adaptDataMCExpr(self._FR.weight()) + ")"
            ## modify cuts to get to control region
            self._mcCorrs = self._mcCorrs + self._FR.cutMods()  + self._FR.mods()
            self._weight = True
        #print "Done creation  %s for task %s in pid %d " % (self._fname, self._name, os.getpid())
    def setScaleFactor(self,scaleFactor):
        self._scaleFactor = scaleFactor
    def getScaleFactor(self):
        return self._scaleFactor
    def setFullYield(self,fullYield):
        self._fullYield = fullYield
    def name(self):
        return self._name
    def cname(self):
        return self._cname
    def hasOption(self,name):
        return (name in self._settings)
    def getOption(self,name,default=None):
        if name in self._settings: return self._settings[name]
        return default
    def setOption(self,name,value):
        self._settings[name] = value
    def adaptDataMCExpr(self,expr):
        ret = expr
        if self._isdata:
            ret = re.sub(r'\$MC\{.*?\}', '', re.sub(r'\$DATA\{(.*?)\}', r'\1', expr));
        else:
            ret = re.sub(r'\$DATA\{.*?\}', '', re.sub(r'\$MC\{(.*?)\}', r'\1', expr));
        return ret
    def adaptExpr(self,expr,cut=False):
        ret = self.adaptDataMCExpr(expr)
        for mcc in self._mcCorrs:
            ret = mcc(ret,self._name,self._cname,cut)
        return ret
    def _init(self):
        if "root://" in self._fname:
            ROOT.gEnv.SetValue("TFile.AsyncReading", 1);
            ROOT.gEnv.SetValue("XNet.Debug", -1); # suppress output about opening connections
            #self._tfile = ROOT.TFile.Open(self._fname+"?readaheadsz=200000") # worse than 65k
            #self._tfile = ROOT.TFile.Open(self._fname+"?readaheadsz=32768") # worse than 65k
            self._tfile = ROOT.TFile.Open(self._fname+"?readaheadsz=65535") # good
            #self._tfile = ROOT.TFile.Open(self._fname+"?readaheadsz=0") #worse than 65k
        else:
            self._tfile = ROOT.TFile.Open(self._fname)
        if not self._tfile: raise RuntimeError, "Cannot open %s\n" % self._fname
        t = self._tfile.Get(self._treename)
        if not t: raise RuntimeError, "Cannot find tree %s in file %s\n" % (self._treename, self._fname)
        self._tree  = t
        #self._tree.SetCacheSize(10*1000*1000)
        if "root://" in self._fname: self._tree.SetCacheSize()
        self._friends = []
        friendOpts = self._options.friendTrees[:]
        friendOpts += (self._options.friendTreesData if self._isdata else self._options.friendTreesMC)
        for tf_tree,tf_file in friendOpts:
            tf = self._tree.AddFriend(tf_tree, tf_file.format(name=self._name, cname=self._cname)),
            self._friends.append(tf)
        self._isInit = True

    def getTree(self):
        if not self._isInit: self._init()
        return self._tree
    def getYields(self,cuts,noEntryLine=False):
        if not self._isInit: self._init()
        report = []; cut = ""
        cutseq = [ ['entry point','1'] ]
        if noEntryLine: cutseq = []
        sequential = False
        if self._options.nMinusOne: 
            cutseq = cuts.nMinusOneCuts()
            cutseq += [ ['all',cuts.allCuts()] ]
            sequential = False
        elif self._options.final:
            cutseq = [ ['all', cuts.allCuts()] ]
        else:
            cutseq += cuts.cuts();
            sequential = True
        for cn,cv in cutseq:
            if sequential:
                if cut: cut += " && "
                cut += "(%s)" % cv
            else:
                cut = cv
            report.append((cn,self._getYield(self._tree,cut)))
        if self._options.fullSampleYields and not noEntryLine:
            report.insert(0, ('full sample', [self._fullYield,0]) )
        return report
    def prettyPrint(self,report):
        # maximum length of the cut descriptions
        clen = max([len(cut) for cut,yields in report]) + 3
        cfmt = "%%-%ds" % clen;

        fmtlen = 12
        nfmtL = "    %8d"
        nfmtS = "    %8.3f" if self._weight else nfmtL

        if self._options.errors:
            nfmtS+=u"%8.3f"
            nfmtL+=u"%8.3f"
            fmtlen+=8
        if self._options.fractions:
            nfmtS+="%7.1f%%"
            nfmtL+="%7.1f%%"
            fmtlen+=8

        print "cut".center(clen),"yield".center(fmtlen)
        print "-"*((fmtlen+1)+clen)
        for i,(cut,(nev,err)) in enumerate(report):
            print cfmt % cut,
            den = report[i-1][1][0] if i>0 else 0
            fraction = nev/float(den) if den > 0 else 1
            if self._options.nMinusOne: 
                fraction = report[-1][1][0]/nev if nev > 0 else 1
            toPrint = (nev,)
            if self._options.errors:    toPrint+=(err,)
            if self._options.fractions: toPrint+=(fraction*100,)
            if self._weight and nev < 1000: print nfmtS % toPrint,
            else                          : print nfmtL % toPrint,
            print ""
    def _getYield(self,tree,cut):
        if self._weight:
            if self._isdata: cut = "(%s)     *(%s)*(%s)" % (self._weightString,                    self._scaleFactor, self.adaptExpr(cut,cut=True))
            else:            cut = "(%s)*(%s)*(%s)*(%s)" % (self._weightString,self._options.lumi, self._scaleFactor, self.adaptExpr(cut,cut=True))
            if self._options.doS2V:
                cut  = scalarToVector(cut)
            ROOT.gROOT.cd()
            if ROOT.gROOT.FindObject("dummy") != None: ROOT.gROOT.FindObject("dummy").Delete()
            histo = ROOT.TH1F("dummy","dummy",1,0.0,1.0); histo.Sumw2()
            nev = tree.Draw("0.5>>dummy", cut, "goff", self._options.maxEntries)
            self.negativeCheck(histo)
            return [ histo.GetBinContent(1), histo.GetBinError(1) ]
        else: 
            cut = self.adaptExpr(cut,cut=True)
            if self._options.doS2V:
                cut  = scalarToVector(cut)
            npass = tree.Draw("1",self.adaptExpr(cut,cut=True),"goff", self._options.maxEntries);
            return [ npass, sqrt(npass) ]
    def _stylePlot(self,plot,spec):
        ## Sample specific-options, from self
        if self.hasOption('FillColor'):
            plot.SetFillColor(self.getOption('FillColor',0))
            plot.SetFillStyle(self.getOption('FillStyle',1001))
        else:
            plot.SetFillStyle(0)
            plot.SetLineWidth(self.getOption('LineWidth',1))
        plot.SetLineColor(self.getOption('LineColor',1))
        plot.SetMarkerColor(self.getOption('MarkerColor',1))
        plot.SetMarkerStyle(self.getOption('MarkerStyle',20))
        plot.SetMarkerSize(self.getOption('MarkerSize',1.6))
        ## Plot specific-options, from spec
        if "TH3" not in plot.ClassName():
            plot.GetYaxis().SetTitle(spec.getOption('YTitle',"Events"))
            plot.GetXaxis().SetTitle(spec.getOption('XTitle',spec.name))
            plot.GetXaxis().SetNdivisions(spec.getOption('XNDiv',510))
    def getPlot(self,plotspec,cut):
        ret = self.getPlotRaw(plotspec.name, plotspec.expr, plotspec.bins, cut, plotspec)
        # fold overflow
        if ret.ClassName() in [ "TH1F", "TH1D" ] :
            n = ret.GetNbinsX()
            if plotspec.getOption('IncludeOverflows',True) and ("TProfile" not in ret.ClassName()):
                ret.SetBinContent(1,ret.GetBinContent(0)+ret.GetBinContent(1))
                ret.SetBinContent(n,ret.GetBinContent(n+1)+ret.GetBinContent(n))
                ret.SetBinError(1,hypot(ret.GetBinError(0),ret.GetBinError(1)))
                ret.SetBinError(n,hypot(ret.GetBinError(n+1),ret.GetBinError(n)))
            rebin = plotspec.getOption('rebinFactor',0)
            if plotspec.bins[0] != "[" and rebin > 1 and n > 5:
                while n % rebin != 0: rebin -= 1
                if rebin != 1: ret.Rebin(rebin)
            if plotspec.getOption('Density',False):
                for b in xrange(1,n+1):
                    ret.SetBinContent( b, ret.GetBinContent(b) / ret.GetXaxis().GetBinWidth(b) )
                    ret.SetBinError(   b, ret.GetBinError(b) / ret.GetXaxis().GetBinWidth(b) )
        self._stylePlot(ret,plotspec)
        return ret
    def getPlotRaw(self,name,expr,bins,cut,plotspec):
        unbinnedData2D = plotspec.getOption('UnbinnedData2D',False) if plotspec != None else False
        profile1D      = plotspec.getOption('Profile1D',False) if plotspec != None else False
        profile2D      = plotspec.getOption('Profile2D',False) if plotspec != None else False
        if self._options.doS2V:
            expr = scalarToVector(expr)
        if not self._isInit: self._init()
        if self._weight:
            if self._isdata: cut = "(%s)     *(%s)*(%s)" % (self._weightString,                    self._scaleFactor, self.adaptExpr(cut,cut=True))
            else:            cut = "(%s)*(%s)*(%s)*(%s)" % (self._weightString,self._options.lumi, self._scaleFactor, self.adaptExpr(cut,cut=True))
        if self._options.doS2V:
            cut  = scalarToVector(cut)
        if ROOT.gROOT.FindObject("dummy") != None: ROOT.gROOT.FindObject("dummy").Delete()
        histo = None
        canKeys = False
        nvars = expr.replace("::","--").count(":")+1
        if nvars == 1 or (nvars == 2 and profile1D):
            if bins[0] == "[":
                edges = [ float(f) for f in bins[1:-1].split(",") ]
                if profile1D: 
                    histo = ROOT.TProfile("dummy","dummy",len(edges)-1,array('f',edges))
                else:
                    histo = ROOT.TH1F("dummy","dummy",len(edges)-1,array('f',edges))
            else:
                (nb,xmin,xmax) = bins.split(",")
                if profile1D:
                    histo = ROOT.TProfile("dummy","dummy",int(nb),float(xmin),float(xmax))
                else:
                    histo = ROOT.TH1F("dummy","dummy",int(nb),float(xmin),float(xmax))
                    canKeys = True
            unbinnedData2D = False
        elif nvars == 2 or (nvars == 3 and profile2D):
            if bins[0] == "[":
                xbins, ybins = bins.split("*")
                xedges = [ float(f) for f in xbins[1:-1].split(",") ]
                yedges = [ float(f) for f in ybins[1:-1].split(",") ]
                if profile2D:
                    histo = ROOT.TProfile2D("dummy","dummy",len(xedges)-1,array('d',xedges),len(yedges)-1,array('d',yedges))
                else:
                    histo = ROOT.TH2F("dummy","dummy",len(xedges)-1,array('f',xedges),len(yedges)-1,array('f',yedges))
            else:
                (nbx,xmin,xmax,nby,ymin,ymax) = bins.split(",")
                if profile2D:
                    histo = ROOT.TProfile2D("dummy","dummy",int(nbx),float(xmin),float(xmax),int(nby),float(ymin),float(ymax))
                    unbinnedData2D = False 
                else:
                    histo = ROOT.TH2F("dummy","dummy",int(nbx),float(xmin),float(xmax),int(nby),float(ymin),float(ymax))
                    unbinnedData2D = (self._name == "data") and unbinnedData2D
        elif nvars == 3:
            ez,ey,ex = [ e.replace("--","::") for e in expr.replace("::","--").split(":") ]
            if bins[0] == "[":
                xbins, ybins, zbins = bins.split("*")
                xedges = [ float(f) for f in xbins[1:-1].split(",") ]
                yedges = [ float(f) for f in ybins[1:-1].split(",") ]
                zedges = [ float(f) for f in zbins[1:-1].split(",") ]
                histo = ROOT.TH3F("dummy","dummy",len(xedges)-1,array('f',xedges),len(yedges)-1,array('f',yedges),len(zedges)-1,array('f',zedges))
            else:
                (nbx,xmin,xmax,nby,ymin,ymax,nbz,zmin,zmax) = bins.split(",")
                histo = ROOT.TH3F("dummy","dummy",int(nbx),float(xmin),float(xmax),int(nby),float(ymin),float(ymax),int(nbz),float(zmin),float(zmax))
            histo.GetXaxis().SetTitle(ex)
            histo.GetYaxis().SetTitle(ey)
            histo.GetZaxis().SetTitle(ez)
        else:
            raise RuntimeError, "Can't make a plot with %d dimensions" % nvars
        histo.Sumw2()
        if unbinnedData2D:
            self._tree.Draw("%s" % (self.adaptExpr(expr)), cut, "", self._options.maxEntries)
            graph = ROOT.gROOT.FindObject("Graph").Clone(name)
            return graph
        drawOpt = "goff"
        if profile1D or profile2D: drawOpt += " PROF";
        self._tree.Draw("%s>>%s" % (self.adaptExpr(expr),"dummy"), cut, drawOpt, self._options.maxEntries)
        if canKeys and histo.GetEntries() > 0 and histo.GetEntries() < self.getOption('KeysPdfMinN',100) and not self._isdata and self.getOption("KeysPdf",False):
            #print "Histogram for %s/%s has %d entries, so will use KeysPdf " % (self._cname, self._name, histo.GetEntries())
            if "/TH1Keys_cc.so" not in ROOT.gSystem.GetLibraries(): 
                ROOT.gROOT.ProcessLine(".L %s/src/CMGTools/TTHAnalysis/python/plotter/TH1Keys.cc+" % os.environ['CMSSW_BASE']);
            (nb,xmin,xmax) = bins.split(",")
            histo = ROOT.TH1KeysNew("dummyk","dummyk",int(nb),float(xmin),float(xmax))
            self._tree.Draw("%s>>%s" % (self.adaptExpr(expr),"dummyk"), cut, "goff", self._options.maxEntries)
            self.negativeCheck(histo)
            return histo.GetHisto().Clone(name)
        #elif not self._isdata and self.getOption("KeysPdf",False):
        #else:
        #    print "Histogram for %s/%s has %d entries, so won't use KeysPdf (%s, %s) " % (self._cname, self._name, histo.GetEntries(), canKeys, self.getOption("KeysPdf",False))
        self.negativeCheck(histo)
        return histo.Clone(name)
    def negativeCheck(self,histo):
        if not self._options.allowNegative: 
            if "TH1" in histo.ClassName():
                for b in xrange(0,histo.GetNbinsX()+2):
                    if histo.GetBinContent(b) < 0: histo.SetBinContent(b, 0.0)
            elif "TH2" in histo.ClassName():
                for bx in xrange(0,histo.GetNbinsX()+2):
                    for by in xrange(0,histo.GetNbinsY()+2):
                        if histo.GetBinContent(bx,by) < 0: histo.SetBinContent(bx,by, 0.0)
            elif "TH3" in histo.ClassName():
                for bx in xrange(0,histo.GetNbinsX()+2):
                    for by in xrange(0,histo.GetNbinsY()+2):
                        for bz in xrange(0,histo.GetNbinsZ()+2):
                            if histo.GetBinContent(bx,by,bz) < 0: histo.SetBinContent(bx,by,bz, 0.0)

    def __str__(self):
        mystr = ""
        mystr += str(self._fname) + '\n'
        mystr += str(self._tfile) + '\n'
        mystr += str(self._weight) + '\n'
        mystr += str(self._scaleFactor)
        return mystr
    def processEvents(self,eventLoop,cut):
        if not self._isInit: self._init()
        cut = self.adaptExpr(cut,cut=True)
        if self._options.doS2V:
            cut  = scalarToVector(cut)
            self._tree.vectorTree = True 
        eventLoop.beginComponent(self)
        eventLoop.loop(self._tree, getattr(self._options, 'maxEvents', -1), cut=cut)
        eventLoop.endComponent(self)

def addTreeToYieldOptions(parser):
    parser.add_option("-l", "--lumi",           dest="lumi",   type="float", default="19.7", help="Luminosity (in 1/fb)");
    parser.add_option("-u", "--unweight",       dest="weight",       action="store_false", default=True, help="Don't use weights (in MC events)");
    parser.add_option("-W", "--weightString",   dest="weightString", type="string", default="1", help="Use weight (in MC events)");
    parser.add_option("--fsy", "--full-sample-yield",  dest="fullSampleYields", action="store_true", default=False, help="Compute also the yield as if all events passed");
    parser.add_option("-f", "--final",  dest="final", action="store_true", help="Just compute final yield after all cuts");
    parser.add_option("-e", "--errors",  dest="errors", action="store_true", help="Include uncertainties in the reports");
    parser.add_option("--tf", "--text-format",   dest="txtfmt", type="string", default="text", help="Output format: text, html");
    parser.add_option("-S", "--start-at-cut",   dest="startCut",   type="string", help="Run selection starting at the cut matched by this regexp, included.") 
    parser.add_option("-U", "--up-to-cut",      dest="upToCut",   type="string", help="Run selection only up to the cut matched by this regexp, included.") 
    parser.add_option("-X", "--exclude-cut", dest="cutsToExclude", action="append", default=[], help="Cuts to exclude (regexp matching cut name), can specify multiple times.") 
    parser.add_option("-I", "--invert-cut",  dest="cutsToInvert",  action="append", default=[], help="Cuts to invert (regexp matching cut name), can specify multiple times.") 
    parser.add_option("-R", "--replace-cut", dest="cutsToReplace", action="append", default=[], nargs=3, help="Cuts to invert (regexp of old cut name, new name, new cut); can specify multiple times.") 
    parser.add_option("-A", "--add-cut",     dest="cutsToAdd",     action="append", default=[], nargs=3, help="Cuts to insert (regexp of cut name after which this cut should go, new name, new cut); can specify multiple times.") 
    parser.add_option("-N", "--n-minus-one", dest="nMinusOne", action="store_true", help="Compute n-minus-one yields and plots")
    parser.add_option("-t", "--tree",          dest="tree", default='treeProducerDarkMatterMonoJet', help="Pattern for tree name");
    parser.add_option("-G", "--no-fractions",  dest="fractions",action="store_false", default=True, help="Don't print the fractions");
    parser.add_option("-F", "--add-friend",    dest="friendTrees",  action="append", default=[], nargs=2, help="Add a friend tree (treename, filename). Can use {name}, {cname} patterns in the treename") 
    parser.add_option("--FMC", "--add-friend-mc",    dest="friendTreesMC",  action="append", default=[], nargs=2, help="Add a friend tree (treename, filename) to MC only. Can use {name}, {cname} patterns in the treename") 
    parser.add_option("--FD", "--add-friend-data",    dest="friendTreesData",  action="append", default=[], nargs=2, help="Add a friend tree (treename, filename) to data trees only. Can use {name}, {cname} patterns in the treename") 
    parser.add_option("--mcc", "--mc-corrections",    dest="mcCorrs",  action="append", default=[], nargs=1, help="Load the following file of mc to data corrections") 
    parser.add_option("--s2v", "--scalar2vector",     dest="doS2V",    action="store_true", default=False, help="Do scalar to vector conversion") 
    parser.add_option("--neg", "--allow-negative-results",     dest="allowNegative",    action="store_true", default=False, help="If the total yield is negative, keep it so rather than truncating it to zero") 
    parser.add_option("--max-entries",     dest="maxEntries", default=1000000000, type="int", help="Max entries to process in each tree") 

def mergeReports(reports):
    import copy
    one = copy.deepcopy(reports[0])
    for i,(c,x) in enumerate(one):
        one[i][1][1] = pow(one[i][1][1], 2)
    for two in reports[1:]:
        for i,(c,x) in enumerate(two):
            one[i][1][0] += x[0]
            one[i][1][1] += pow(x[1],2)
    for i,(c,x) in enumerate(one):
        one[i][1][1] = sqrt(one[i][1][1])
    return one

def mergePlots(name,plots):
    one = plots[0].Clone(name)
    if "TGraph" in one.ClassName():
        others = ROOT.TList()
        for two in plots[1:]: 
            others.Add(two)
        one.Merge(others)
    else:         
        for two in plots[1:]: 
            one.Add(two)
    return one

