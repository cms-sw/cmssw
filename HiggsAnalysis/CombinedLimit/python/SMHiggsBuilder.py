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
    def makeScaling(self,what, Cb='Cb', Ctop='Ctop', CW='CW', CZ='CZ', Ctau='Ctau'):
        prefix = 'SM_%(what)s_' % locals()
#        self.modelBuilder.doVar('One[1]')
#        self.modelBuilder.doVar('Zero[0]') 
        if what == 'qqH':
            for sqrts in ('7TeV', '8TeV'):
                rooName = prefix+'RVBF_'+sqrts
                self.textToSpline(rooName, os.path.join(self.coupPath, 'R_VBF_%(sqrts)s.txt'%locals()), ycol=1 )
                scalingName = 'Scaling_'+what+'_'+sqrts
#                print 'Building '+ scalingName
                rooExpr = 'expr::%(scalingName)s("(@0+ @1 * @2 )/(1+@2) ", %(CW)s, %(CZ)s, %(rooName)s)'%locals()
#                print  rooExpr
                self.modelBuilder.factory_(rooExpr)
        elif what == 'ggH':
            structure = {'sigma_tt':2, 'sigma_bb':3, 'sigma_tb':4}
            for sqrts in ('7TeV', '8TeV'):
                for qty, column in structure.iteritems():
                    rooName = prefix+qty+'_'+sqrts
                    self.textToSpline(rooName, os.path.join(self.coupPath, 'ggH_%(sqrts)s.txt'%locals()), ycol=column )
                scalingName = 'Scaling_'+what+'_'+sqrts
#                print 'Building '+scalingName
                rooExpr = 'expr::%(scalingName)s("(@0*@0)*@2  + (@1*@1)*@3 + (@0*@1)*@4", %(Ctop)s, %(Cb)s, %(prefix)ssigma_tt_%(sqrts)s, %(prefix)ssigma_bb_%(sqrts)s, %(prefix)ssigma_tb_%(sqrts)s)'%locals()
#                print  rooExpr
                self.modelBuilder.factory_(rooExpr)
        elif what == 'hgluglu':
            structure = {'Gamma_tt':2, 'Gamma_bb':3, 'Gamma_tb':4}
            for qty, column in structure.iteritems():
                rooName = prefix+qty
                self.textToSpline(rooName, os.path.join(self.coupPath, 'Gamma_Hgluongluon.txt'), ycol=column )
            scalingName = 'Scaling_'+what
#            print 'Building '+scalingName
            rooExpr = 'expr::%(scalingName)s("(@0*@0)*@2  + (@1*@1)*@3 + (@0*@1)*@4", %(Ctop)s, %(Cb)s, %(prefix)sGamma_tt, %(prefix)sGamma_bb, %(prefix)sGamma_tb)'%locals()
#            print  rooExpr
            self.modelBuilder.factory_(rooExpr)
        elif what in ['hgg', 'hZg']:
            fileFor = {'hgg':'Gamma_Hgammagamma.txt',
                       'hZg':'Gamma_HZgamma.txt'}
            structure = {'Gamma_tt':2, 'Gamma_bb':3, 'Gamma_WW':4,
                         'Gamma_tb':5, 'Gamma_tW':6, 'Gamma_bW':7,
                         'Gamma_ll':8,
                         'Gamma_tl':9, 'Gamma_bl':10, 'Gamma_lW':11}
            for qty, column in structure.iteritems():
                rooName = prefix+qty
                self.textToSpline(rooName, os.path.join(self.coupPath, fileFor[what]), ycol=column )
            scalingName = 'Scaling_'+what
#            print 'Building '+scalingName
            rooExpr = 'expr::%(scalingName)s(\
"(@0*@0)*@4  + (@1*@1)*@5 + (@2*@2)*@6 + (@0*@1)*@7 + (@0*@2)*@8 + (@1*@2)*@9 + (@3*@3)*@10 + (@0*@3)*@11 + (@1*@3)*@12 + (@2*@3)*@13",\
%(Ctop)s, %(Cb)s, %(CW)s, %(Ctau)s,\
%(prefix)sGamma_tt, %(prefix)sGamma_bb, %(prefix)sGamma_WW,\
%(prefix)sGamma_tb, %(prefix)sGamma_tW, %(prefix)sGamma_bW,\
%(prefix)sGamma_ll,\
%(prefix)sGamma_tl, %(prefix)sGamma_bl, %(prefix)sGamma_lW\
)'%locals()
#            print  rooExpr
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
