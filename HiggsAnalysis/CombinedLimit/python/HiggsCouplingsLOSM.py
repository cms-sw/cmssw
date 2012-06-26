from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class CvCfHiggsLOSM(SMLikeHiggsModel):
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
        self.modelBuilder.doVar("CF[1,-2,2]")
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
        #self.doDebugDump()
        self.setup()
    def setup(self):
        ## Add some common ingredients
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        self.SMH.textToSpline( 'mb', os.path.join(datadir, 'running_constants.txt'), ycol=2 );
        mb = self.modelBuilder.out.function('mb')
        mH = self.modelBuilder.out.function('MH')
        CF = self.modelBuilder.out.function('CF')
        CV = self.modelBuilder.out.function('CV')

        RHggCvCf = ROOT.RooScaleHGamGamLOSM('CvCf_cgamma', 'LO SM Hgamgam scaling', mH, CF, CV, mb, CF)
        self.modelBuilder.out._import(RHggCvCf)
        #Rgluglu = ROOT.RooScaleHGluGluLOSM('Rgluglu', 'LO SM Hgluglu scaling', mH, CF, mb, CF)
        #self.modelBuilder.out._import(Rgluglu)
        
        ## partial witdhs, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_('expr::CvCf_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::CvCf_Gscal_sumv("@0*@0 * (@1+@2+@3)", CV, SM_BR_hww, SM_BR_hzz, SM_BR_hZg)') 
        self.modelBuilder.factory_('expr::CvCf_Gscal_gg("@0*@0 * @1", CvCf_cgamma, SM_BR_hgg)') 
        self.modelBuilder.factory_('sum::CvCf_Gscal_tot(CvCf_Gscal_sumf, CvCf_Gscal_sumv, CvCf_Gscal_gg)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::CvCf_BRscal_hgg("@0*@0/@1", CvCf_cgamma, CvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCf_BRscal_hf("@0*@0/@1", CF, CvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::CvCf_BRscal_hv("@0*@0/@1", CV, CvCf_Gscal_tot)')
        
        self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "CvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = 'CF' if production in ["ggH","ttH"] else 'CV'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = "hf"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name


cVcF = CvCfHiggsLOSM()
