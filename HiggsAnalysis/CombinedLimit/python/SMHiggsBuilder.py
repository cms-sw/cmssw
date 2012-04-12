from math import *
from array import array
import os 
import ROOT

class SMHiggsBuilder:
    def __init__(self,modelBuilder,datadir=None):
        self.modelBuilder = modelBuilder
        if datadir == None:
            datadir = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/"
        self.datadir = datadir
    def makeXS(self,process):
        if process == "ggH": self.textToSpline("SM_XS_ggH", self.datadir+"YR-XS-ggH.txt");
        if process == "qqH": self.textToSpline("SM_XS_qqH", self.datadir+"YR-XS-vbfH.txt");
        if process == "WH":  self.textToSpline("SM_XS_WH", self.datadir+"YR-XS-WH.txt");
        if process == "ZH":  self.textToSpline("SM_XS_ZH", self.datadir+"YR-XS-ZH.txt");
        if process == "VH":  
            makeXS("WH"); makeXS("ZH");
            self.modelBuilder.factory_('sum::SM_XS_VH(SM_XS_WH,SM_XS_ZH)')
    def makeTotalWidth(self):
        self.textToSpline("SM_GammaTot", self.datadir+"YR-BR.txt", ycol=6);
    def makeBR(self,decay):
        if decay == "hww": self.textToSpline("SM_BR_hww", self.datadir+"YR-BR.txt", ycol=4);
        if decay == "hzz": self.textToSpline("SM_BR_hzz", self.datadir+"YR-BR.txt", ycol=5);
        if decay == "hgg": self.textToSpline("SM_BR_hgg", self.datadir+"YR-BR.txt", ycol=2);
        if decay == "hZg": self.textToSpline("SM_BR_hZg", self.datadir+"YR-BR.txt", ycol=3);
        if decay == "hbb": self.textToSpline("SM_BR_hbb", self.datadir+"YR-BR1.txt", ycol=1);
        if decay == "htt": self.textToSpline("SM_BR_htt", self.datadir+"YR-BR1.txt", ycol=2);
        if decay == "hmm": self.textToSpline("SM_BR_hmm", self.datadir+"YR-BR1.txt", ycol=3);
        if decay == "hss": self.textToSpline("SM_BR_hss", self.datadir+"YR-BR1.txt", ycol=4);
        if decay == "hcc": self.textToSpline("SM_BR_hcc", self.datadir+"YR-BR1.txt", ycol=5);
        if decay == "hgluglu": self.textToSpline("SM_BR_hgluglu", self.datadir+"YR-BR.txt",   ycol=1);
        if decay == "htoptop": self.textToSpline("SM_BR_htoptop", self.datadir+"YR-BR1.txt", ycol=6);
    def makePartialWidth(self,decay):
        self.makeTotalWidth(); 
        self.makeBR(decay);
        self.modelBuilder.factory_('prod::SM_Gamma_%s(SM_GammaTot,SM_BR_%s)' % (decay,decay))
    def dump(self,name,xvar,values,logfile):
        xv = self.modelBuilder.out.var(xvar)
        yf = self.modelBuilder.out.function(name)
        if yf == None: raise RuntimeError, "Missing "+name
        log = open(logfile, "w")
        for x in values:
            xv.setVal(x)
            log.write("%.3f\t%.7g\n" % (x, yf.getVal()) )
    def textToSpline(self,name,filename,xvar="MH",ycol=1,xcol=0,skipRows=1,algo="CSPLINE"):
        if (self.modelBuilder.out.function(name) != None): return
        x = []; y = []
        file = open(filename,'r')
        lines = [l for l in file]
        for line in lines[skipRows:]:
            if len(line.strip()) == 0: continue
            cols = line.split();
            x.append(float(cols[xcol]))
            y.append(float(cols[ycol]))
        xv = self.modelBuilder.out.var(xvar)
        spline = ROOT.RooSpline1D(name, "file %s, x=%d, y=%d" % (filename,xcol,ycol), xv, len(x), array('d', x), array('d', y), algo)
        self.modelBuilder.out._import(spline)
