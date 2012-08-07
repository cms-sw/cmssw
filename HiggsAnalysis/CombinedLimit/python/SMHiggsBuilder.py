from math import *
from array import array
import os 
import ROOT

class SMHiggsBuilder:
    def __init__(self,modelBuilder,datadir=None):
        self.modelBuilder = modelBuilder
        if datadir == None:
            datadir = os.environ['CMSSW_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg"
        self.datadir = datadir
        self.brpath = os.path.join(self.datadir,'sm/br')
        self.coupPath = os.path.join(self.datadir,'couplings')

    def makeXS(self,process, energy='7TeV'):
        self.xspath = os.path.join(self.datadir, 'sm/xs', energy)
        if process == "ggH": self.textToSpline("SM_XS_ggH_"+energy, os.path.join(self.xspath, energy+"-ggH.txt") );
        if process == "qqH": self.textToSpline("SM_XS_qqH_"+energy, os.path.join(self.xspath, energy+"-vbfH.txt") );
        if process == "ttH": self.textToSpline("SM_XS_ttH_"+energy, os.path.join(self.xspath, energy+"-ttH.txt") );
        if process == "WH":  self.textToSpline("SM_XS_WH_"+energy,  os.path.join(self.xspath, energy+"-WH.txt") );
        if process == "ZH":  self.textToSpline("SM_XS_ZH_"+energy,  os.path.join(self.xspath, energy+"-ZH.txt") );
        if process == "VH":  
            makeXS("WH", energy); makeXS("ZH", energy);
            self.modelBuilder.factory_('sum::SM_XS_VH_'+energy+'(SM_XS_WH_'+energy+',SM_XS_ZH_'+energy+')')
    def makeTotalWidth(self):
        self.textToSpline("SM_GammaTot", os.path.join(self.brpath,"BR.txt"), ycol=6);
    def makeBR(self,decay):
        if decay == "hww": self.textToSpline("SM_BR_hww", os.path.join(self.brpath, "BR.txt"), ycol=4);
        if decay == "hzz": self.textToSpline("SM_BR_hzz", os.path.join(self.brpath, "BR.txt"), ycol=5);
        if decay == "hgg": self.textToSpline("SM_BR_hgg", os.path.join(self.brpath, "BR.txt"), ycol=2);
        if decay == "hZg": self.textToSpline("SM_BR_hZg", os.path.join(self.brpath, "BR.txt"), ycol=3);
        if decay == "hbb": self.textToSpline("SM_BR_hbb", os.path.join(self.brpath, "BR1.txt"), ycol=1);
        if decay == "htt": self.textToSpline("SM_BR_htt", os.path.join(self.brpath, "BR1.txt"), ycol=2);
        if decay == "hmm": self.textToSpline("SM_BR_hmm", os.path.join(self.brpath, "BR1.txt"), ycol=3);
        if decay == "hss": self.textToSpline("SM_BR_hss", os.path.join(self.brpath, "BR1.txt"), ycol=4);
        if decay == "hcc": self.textToSpline("SM_BR_hcc", os.path.join(self.brpath, "BR1.txt"), ycol=5);
        if decay == "hgluglu": self.textToSpline("SM_BR_hgluglu", os.path.join(self.brpath, "BR.txt"),   ycol=1);
        if decay == "htoptop": self.textToSpline("SM_BR_htoptop", os.path.join(self.brpath, "BR1.txt"), ycol=6);
    def makePartialWidth(self,decay):
        self.makeTotalWidth(); 
        self.makeBR(decay);
        self.modelBuilder.factory_('prod::SM_Gamma_%s(SM_GammaTot,SM_BR_%s)' % (decay,decay))
    def makeScaling(self,what, Cb='Cb', Ctop='Ctop', CW='CW', CZ='CZ', Ct='Ct'):
        prefix = 'SM_%(what)s_' % locals()
        self.modelBuilder.doVar('One[1]') 
        if what == 'qqH':
            for sqrts in ('7TeV', '8TeV'):
                rooName = prefix+'RVBF_'+sqrts
                print 'Building '+rooName
                self.textToSpline(rooName,
                                  os.path.join(self.coupPath, 'R_VBF_%(sqrts)s.txt'%locals()),
                                  ycol=1 )
                scalingName = 'Scaling_'+what+'_'+sqrts
                print 'Building '+ scalingName
                rooExpr = 'expr::%(scalingName)s("(@0+ @1 * @2 )/(1+@2) ", %(CW)s, %(CZ)s, %(rooName)s)'%locals()
                print  rooExpr
                self.modelBuilder.factory_(rooExpr)
        elif what == 'ggH':
            structure = {'sigma_tt':1, 'sigma_bb':2, 'sigma_tb':3}
            for sqrts in ('7TeV', '8TeV'):
                for qty, column in structure.iteritems():
                    rooName = prefix+qty+'_'+sqrts
                    print 'Building '+rooName
                    #
                scalingName = 'Scaling_'+what+'_'+sqrts
                print 'Building '+scalingName
                rooExpr = 'expr::%(scalingName)s("@0", One)'%locals()
                print  rooExpr
                self.modelBuilder.factory_(rooExpr)
        elif what == 'hgluglu':
            structure = {'Gamma_tt':1, 'Gamma_bb':2, 'Gamma_tb':3}
            for qty, column in structure.iteritems():
                rooName = prefix+qty
                print 'Building '+rooName
                #
            scalingName = 'Scaling_'+what
            print 'Building '+scalingName
            rooExpr = 'expr::%(scalingName)s("@0", One)'%locals()
            print  rooExpr
            self.modelBuilder.factory_(rooExpr)
        elif what == 'hgg':
            structure = {'Gamma_tt':1, 'Gamma_bb':2, 'Gamma_tb':3}
            for qty, column in structure.iteritems():
                rooName = prefix+qty
                print 'Building '+rooName
            scalingName = 'Scaling_'+what
            print 'Building '+scalingName
            rooExpr = 'expr::%(scalingName)s("@0", One)'%locals()
            print  rooExpr
            self.modelBuilder.factory_(rooExpr)
        else:
            raise RuntimeError, "There is no scaling defined for %(what)s" % locals()
                
        
            
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

#if __name__ == "__main__":
#   sm = SMHiggsBuilder()
