from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class CvCfHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
        self.cVRange = ['0','2']
        self.cFRange = ['-2','2']
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
            if po.startswith("cVRange="):
                self.cVRange = po.replace("cVRange=","").split(":")
                if len(self.cVRange) != 2:
                    raise RuntimeError, "cV signal strength range requires minimal and maximal value"
                elif float(self.cVRange[0]) >= float(self.cVRange[1]):
                    raise RuntimeError, "minimal and maximal range swapped. Second value must be larger first one"
            if po.startswith("cFRange="):
                self.cFRange = po.replace("cFRange=","").split(":")
                if len(self.cFRange) != 2:
                    raise RuntimeError, "cF signal strength range requires minimal and maximal value"
                elif float(self.cFRange[0]) >= float(self.cFRange[1]):
                    raise RuntimeError, "minimal and maximal range swapped. Second value must be larger first one"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("CV[1,%s,%s]" % (self.cVRange[0], self.cVRange[1]))
        self.modelBuilder.doVar("CF[1,%s,%s]" % (self.cVRange[0], self.cVRange[1]))
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'CV,CF,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'CV,CF')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):
        self.decayScaling = {
            'hgg':'hgg',
            'hZg':'hZg',
            'hww':'hvv',
            'hzz':'hvv',
            'hbb':'hff',
            'htt':'hff',
            }
        self.productionScaling = {
            'ggH':'CF',
            'ttH':'CF',
            'qqH':'CV',
            'WH':'CV',
            'ZH':'CV',
            'VH':'CV',
            }
        
        self.SMH.makeScaling('hgg', Cb='CF', Ctop='CF', CW='CV', Ctau='CF')
        self.SMH.makeScaling('hZg', Cb='CF', Ctop='CF', CW='CV', Ctau='CF')
        
        ## partial widths, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::CvCf_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::CvCf_Gscal_sumv("@0*@0 * (@1+@2)", CV, SM_BR_hww, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::CvCf_Gscal_gg("@0 * @1", Scaling_hgg, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::CvCf_Gscal_Zg("@0 * @1", Scaling_hZg, SM_BR_hZg)') 
        self.modelBuilder.factory_('sum::CvCf_Gscal_tot(CvCf_Gscal_sumf, CvCf_Gscal_sumv, CvCf_Gscal_gg, CvCf_Gscal_Zg)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::CvCf_BRscal_hgg("@0/@1", Scaling_hgg, CvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCf_BRscal_hZg("@0/@1", Scaling_hZg, CvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCf_BRscal_hff("@0*@0/@1", CF, CvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCf_BRscal_hvv("@0*@0/@1", CV, CvCf_Gscal_tot)')
        
        self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):

        name = "CvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name):
            return name
        
        XSscal = self.productionScaling[production]
        BRscal = self.decayScaling[decay]
        self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name

class CvCfXgHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("CV[1,0.0,1.5]")
        self.modelBuilder.doVar("CF[1,-1.5,1.5]")
        self.modelBuilder.doVar("XG[0,-4,4]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'CV,CF,XG,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'CV,CF,XG')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
    def setup(self):
        ## Add some common ingredients
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        self.SMH.textToSpline( 'mb', os.path.join(datadir, 'running_constants.txt'), ycol=2 );
        mb = self.modelBuilder.out.function('mb')
        mH = self.modelBuilder.out.var('MH')
        CF = self.modelBuilder.out.var('CF')
        CV = self.modelBuilder.out.var('CV')
        XG = self.modelBuilder.out.var('XG')

        RHggCvCfXg = ROOT.RooScaleHGamGamLOSMPlusX('CvCfXg_cgammaSq', 'LO SM Hgamgam scaling', mH, CF, CV, mb, CF, XG)
        self.modelBuilder.out._import(RHggCvCfXg)
        #Rgluglu = ROOT.RooScaleHGluGluLOSMPlusX('Rgluglu', 'LO SM Hgluglu scaling', mH, CF, mb, CF)
        #self.modelBuilder.out._import(Rgluglu)
        
        ## partial witdhs, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::CvCfXg_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::CvCfXg_Gscal_sumv("@0*@0 * (@1+@2+@3)", CV, SM_BR_hww, SM_BR_hzz, SM_BR_hZg)') 
        self.modelBuilder.factory_('expr::CvCfXg_Gscal_gg("@0 * @1", CvCfXg_cgammaSq, SM_BR_hgg)') 
        self.modelBuilder.factory_('sum::CvCfXg_Gscal_tot(CvCfXg_Gscal_sumf, CvCfXg_Gscal_sumv, CvCfXg_Gscal_gg)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::CvCfXg_BRscal_hgg("@0/@1", CvCfXg_cgammaSq, CvCfXg_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCfXg_BRscal_hf("@0*@0/@1", CF, CvCfXg_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCfXg_BRscal_hv("@0*@0/@1", CV, CvCfXg_Gscal_tot)')
        
        self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "CvCfXg_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = 'CF' if production in ["ggH","ttH"] else 'CV'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = "hf"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CvCfXg_BRscal_%s)' % (name, XSscal, BRscal))
        return name

class CfXgHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("CV[1]")
        self.modelBuilder.doVar("CF[1,-1.5,1.5]")
        self.modelBuilder.doVar("XG[0,-4,4]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'CF,XG,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'CF,XG')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
    def setup(self):
        ## Add some common ingredients
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        self.SMH.textToSpline( 'mb', os.path.join(datadir, 'running_constants.txt'), ycol=2 );
        mb = self.modelBuilder.out.function('mb')
        mH = self.modelBuilder.out.var('MH')
        CF = self.modelBuilder.out.var('CF')
        CV = self.modelBuilder.out.var('CV')
        XG = self.modelBuilder.out.var('XG')

        RHggCfXg = ROOT.RooScaleHGamGamLOSMPlusX('CfXg_cgammaSq', 'LO SM Hgamgam scaling', mH, CF, CV, mb, CF, XG)
        self.modelBuilder.out._import(RHggCfXg)
        #Rgluglu = ROOT.RooScaleHGluGluLOSMPlusX('Rgluglu', 'LO SM Hgluglu scaling', mH, CF, mb, CF)
        #self.modelBuilder.out._import(Rgluglu)
        
        ## partial witdhs, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::CfXg_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('sum::CfXg_Gscal_sumv(SM_BR_hww, SM_BR_hzz, SM_BR_hZg)') 
        self.modelBuilder.factory_('expr::CfXg_Gscal_gg("@0 * @1", CfXg_cgammaSq, SM_BR_hgg)') 
        self.modelBuilder.factory_('sum::CfXg_Gscal_tot(CfXg_Gscal_sumf, CfXg_Gscal_sumv, CfXg_Gscal_gg)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::CfXg_BRscal_hgg("@0/@1", CfXg_cgammaSq, CfXg_Gscal_tot)')
        self.modelBuilder.factory_('expr::CfXg_BRscal_hf("@0*@0/@1", CF, CfXg_Gscal_tot)')
        self.modelBuilder.factory_('expr::CfXg_BRscal_hv("1.0/@0", CfXg_Gscal_tot)')
        
        self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "CfXg_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = 'CF' if production in ["ggH","ttH"] else 'CV'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = "hf"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CfXg_BRscal_%s)' % (name, XSscal, BRscal))
        return name

