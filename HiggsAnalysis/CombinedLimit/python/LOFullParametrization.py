from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os


class C5(SMLikeHiggsModel):
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
        self.modelBuilder.doVar("kV[1,0.0,2.0]")
        self.modelBuilder.doVar("ktau[1,0.0,2.0]")
        self.modelBuilder.doVar("kquark[1,0.0,2.0]")
        self.modelBuilder.doVar("kgluon[1,0.0,2.0]")
        self.modelBuilder.doVar("kgamma[1,0.0,2.0]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kV,ktau,kquark,kgluon,kgamma,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kV,ktau,kquark,kgluon,kgamma')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]: self.SMH.makeBR(d)

        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::c6_Gscal_Vectors("@0*@0 * (@1+@2)", kV, SM_BR_hzz, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::c6_Gscal_tau("@0*@0 * (@1+@2)", ktau, SM_BR_htt, SM_BR_hmm)') 
        self.modelBuilder.factory_('expr::c6_Gscal_quark("@0*@0 * (@1+@2+@3+@4)", kquark, SM_BR_htoptop, SM_BR_hcc,SM_BR_hbb, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::c6_Gscal_gluon("@0*@0 * @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::c6_Gscal_gamma("@0*@0 * (@1+@2)", kgamma, SM_BR_hgg, SM_BR_hZg)')
        self.modelBuilder.factory_('sum::c6_Gscal_tot(c6_Gscal_Vectors, c6_Gscal_tau, c6_Gscal_quark, c6_Gscal_gluon, c6_Gscal_gamma)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::c6_BRscal_hvv("@0*@0/@1", kV, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_htt("@0*@0/@1", ktau, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_hbb("@0*@0/@1", kquark, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_hgg("@0*@0/@1", kgamma, c6_Gscal_tot)')

    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "c6_XSBRscal_%s_%s" % (production,decay)
        print '[LOFullParametrization::C6]'
        print name, production, decay, energy
        if self.modelBuilder.out.function(name) == None:
            XSscal = "kgluon"
            if production in ["WH","ZH","VH","qqH"]: XSscal = "kV"
            if production == "ttH": XSscal = "kquark"
            BRscal = "hgg"
            if decay in ["hbb", "htt"]: BRscal = decay
            if decay in ["hww", "hzz"]: BRscal = "hvv"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, c6_BRscal_%s)' % (name, XSscal, BRscal))
        return name


class C6(SMLikeHiggsModel):
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
        self.modelBuilder.doVar("kV[1,0.0,2.0]")
        self.modelBuilder.doVar("ktau[1,0.0,2.0]")
        self.modelBuilder.doVar("ktop[1,0.0,2.0]")
        self.modelBuilder.doVar("kbottom[1,0.0,2.0]")
        self.modelBuilder.doVar("kgluon[1,0.0,2.0]")
        self.modelBuilder.doVar("kgamma[1,0.0,2.0]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'kV,ktau,ktop,kbottom,kgluon,kgamma,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'kV,ktau,ktop,kbottom,kgluon,kgamma')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):

        # SM BR
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]: self.SMH.makeBR(d)

        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::c6_Gscal_Vectors("@0*@0 * (@1+@2)", kV, SM_BR_hzz, SM_BR_hww)') 
        self.modelBuilder.factory_('expr::c6_Gscal_tau("@0*@0 * (@1+@2)", ktau, SM_BR_htt, SM_BR_hmm)') 
        self.modelBuilder.factory_('expr::c6_Gscal_top("@0*@0 * (@1+@2)", ktop, SM_BR_htoptop, SM_BR_hcc)')
        self.modelBuilder.factory_('expr::c6_Gscal_bottom("@0*@0 * (@1+@2)", kbottom, SM_BR_hbb, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::c6_Gscal_gluon("@0*@0 * @1", kgluon, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::c6_Gscal_gamma("@0*@0 * (@1+@2)", kgamma, SM_BR_hgg, SM_BR_hZg)')
        self.modelBuilder.factory_('sum::c6_Gscal_tot(c6_Gscal_Vectors, c6_Gscal_tau, c6_Gscal_top, c6_Gscal_bottom, c6_Gscal_gluon, c6_Gscal_gamma)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_('expr::c6_BRscal_hvv("@0*@0/@1", kV, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_htt("@0*@0/@1", ktau, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_hbb("@0*@0/@1", kbottom, c6_Gscal_tot)')
        self.modelBuilder.factory_('expr::c6_BRscal_hgg("@0*@0/@1", kgamma, c6_Gscal_tot)')

    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "c6_XSBRscal_%s_%s" % (production,decay)
        print '[LOFullParametrization::C6]'
        print name, production, decay, energy
        if self.modelBuilder.out.function(name) == None:
            XSscal = "kgluon"
            if production in ["WH","ZH","VH","qqH"]: XSscal = "kV" 
            if production == "ttH": XSscal = "ktop"
            BRscal = "hgg"
            if decay in ["hbb", "htt"]: BRscal = decay
            if decay in ["hww", "hzz"]: BRscal = "hvv"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, c6_BRscal_%s)' % (name, XSscal, BRscal))
        return name



