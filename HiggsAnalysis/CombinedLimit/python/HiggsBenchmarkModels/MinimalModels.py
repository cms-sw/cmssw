from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class HiggsMinimal(SMLikeHiggsModel):
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
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrama for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        self.modelBuilder.doVar("kgluon[1,0,2]")
        self.modelBuilder.doVar("kgamma[1,0,3]")
        self.modelBuilder.doVar("kV[1,0,3]")
        self.modelBuilder.doVar("kf[1,0,3]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma,kV,kf,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma,kV,kf')
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
            'ggH':'kgluon',
            'ttH':'kf',
            'qqH':'kV',
            'WH':'kV',
            'ZH':'kV',
            'VH':'kV',
        }

        self.SMH.makeScaling('hZg', Cb='kf', Ctop='kf', CW='kV', Ctau='kf')
        
        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)

        ## total witdh, normalized to the SM one
        self.modelBuilder.factory_('expr::minimal_Gscal_gg("@0*@0 * @1", kgamma, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::minimal_Gscal_gluglu("@0*@0 * @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::minimal_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6)", kf, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::minimal_Gscal_sumv("@0*@0 * (@1+@2)", kV, SM_BR_hww, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::minimal_Gscal_Zg("@0 * @1", Scaling_hZg, SM_BR_hZg)') 
        
        self.modelBuilder.factory_('sum::minimal_Gscal_tot(minimal_Gscal_sumf, minimal_Gscal_sumv, minimal_Gscal_Zg, minimal_Gscal_gg, minimal_Gscal_gluglu)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::minimal_BRscal_hgg("@0*@0/@1", kgamma, minimal_Gscal_tot)')
        self.modelBuilder.factory_('expr::minimal_BRscal_hZg("@0/@1", Scaling_hZg, minimal_Gscal_tot)')
        self.modelBuilder.factory_('expr::minimal_BRscal_hff("@0*@0/@1", kf, minimal_Gscal_tot)')
        self.modelBuilder.factory_('expr::minimal_BRscal_hvv("@0*@0/@1", kV, minimal_Gscal_tot)')

        # verbosity
        #self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):
        
        name = "minimal_XSBRscal_%s_%s" % (production,decay)

        if self.modelBuilder.out.function(name):
            return name
        
        XSscal = self.productionScaling[production]
        BRscal = self.decayScaling[decay]
        self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, minimal_BRscal_%s)' % (name, XSscal, BRscal))

        return name
