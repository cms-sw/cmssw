from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class HiggsLoops(SMLikeHiggsModel):
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
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        self.decayScaling = {
            'hgg':'hgg',
            'hZg':'hxx',
            'hww':'hxx',
            'hzz':'hxx',
            'hbb':'hxx',
            'htt':'hxx',
            }
        
        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)

        ## total witdh, normalized to the SM one
        self.modelBuilder.factory_('sum::loopGluonGamma_Gscal_OtherDecays(SM_BR_hbb, SM_BR_htt, SM_BR_hmm, SM_BR_hss, SM_BR_hzz, SM_BR_hww, SM_BR_hcc, SM_BR_htoptop, SM_BR_hZg)')
        self.modelBuilder.factory_('expr::loopGluonGamma_Gscal_gg("@0*@0* @1", kgamma, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::loopGluonGamma_Gscal_gluglu("@0*@0* @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('sum::loopGluonGamma_Gscal_tot(loopGluonGamma_Gscal_OtherDecays, loopGluonGamma_Gscal_gg, loopGluonGamma_Gscal_gluglu)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::loopGluonGamma_BRscal_hxx("1.0/@0",loopGluonGamma_Gscal_tot)')
        self.modelBuilder.factory_('expr::loopGluonGamma_BRscal_hgg("@0*@0/@1", kgamma, loopGluonGamma_Gscal_tot)')

        # verbosity
        #self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):
        
        name = "loopGluonGamma_XSBRscal_%s_%s" % (production,decay)

        if self.modelBuilder.out.function(name):
            return name
        
        BRscal = self.decayScaling[decay]

        if production == "ggH":
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", kgluon, loopGluonGamma_BRscal_%s)' % (name, BRscal))
        else:
            self.modelBuilder.factory_('expr::%s("1.0 * @0", loopGluonGamma_BRscal_%s)' % (name,BRscal))
                
        return name


class HiggsLoopsInvisible(SMLikeHiggsModel):
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
        self.modelBuilder.doVar("BRInvUndet[0,0,1]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma,BRInvUndet,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kgluon,kgamma,BRInvUndet')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        self.decayScaling = {
            'hgg':'hgg',
            'hZg':'hxx',
            'hww':'hxx',
            'hzz':'hxx',
            'hbb':'hxx',
            'htt':'hxx',
            }
        
        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)

        ## total witdh, normalized to the SM one
        self.modelBuilder.factory_('sum::loopGluonGamma_Gscal_OtherDecays(SM_BR_hbb, SM_BR_htt, SM_BR_hmm, SM_BR_hss, SM_BR_hzz, SM_BR_hww, SM_BR_hcc, SM_BR_htoptop, SM_BR_hZg)')
        self.modelBuilder.factory_('expr::loopGluonGamma_Gscal_gg("@0*@0* @1", kgamma, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::loopGluonGamma_Gscal_gluglu("@0*@0* @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::loopGluonGamma_Gscal_tot("(@1+@2+@3)/(1.0-@0)", BRInvUndet, loopGluonGamma_Gscal_OtherDecays, loopGluonGamma_Gscal_gg, loopGluonGamma_Gscal_gluglu)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::loopGluonGamma_BRscal_hxx("1.0/@0",loopGluonGamma_Gscal_tot)')
        self.modelBuilder.factory_('expr::loopGluonGamma_BRscal_hgg("@0*@0/@1", kgamma, loopGluonGamma_Gscal_tot)')

        # verbosity
        #self.modelBuilder.out.Print()

    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "loopGluonGamma_XSBRscal_%s_%s" % (production,decay)

        if self.modelBuilder.out.function(name):
            return name

        BRscal = self.decayScaling[decay]

        if production == "ggH":
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", kgluon, loopGluonGamma_BRscal_%s)' % (name, BRscal))
        else:
            self.modelBuilder.factory_('expr::%s("1.0 * @0", loopGluonGamma_BRscal_%s)' % (name, BRscal))
                
        return name
