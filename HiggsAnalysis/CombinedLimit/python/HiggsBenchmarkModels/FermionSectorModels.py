from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os


class C5qlHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
        self.universalCF = False
        self.fix = []
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po == "universalCF": universalCF = True
            if po.startswith("fix="): self.fix = po.replace("fix=","").split(",")
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
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("Cg[1,0,10]")
        self.modelBuilder.doVar("Cv[1,0,10]")
        self.modelBuilder.doVar("Cglu[1,0,10]")
        POI = "Cg,Cv,Cglu"
        if self.universalCF:
            self.modelBuilder.doVar("Cf[1,0,10]")
            POI += ",Cf"
        else:
            self.modelBuilder.doVar("Cq[1,0,10]")
            self.modelBuilder.doVar("Cl[1,0,10]")
            POI += ",Cq,Cl"
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            POI += ",MH"
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
        for F in self.fix:
           self.modelBuilder.out.var(F).setConstant(True)
           if F+"," in POI: POI = POI.replace(F+",", "")
           else:            POI = POI.replace(","+F, "")
        self.modelBuilder.doSet("POI",POI)
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::C5ql_Gscal_sumglu("@0*@0 * @1", Cglu, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::C5ql_Gscal_sumg("@0*@0 * @1", Cg, SM_BR_hgg)')
        self.modelBuilder.factory_('expr::C5ql_Gscal_sumv("@0*@0 * (@1+@2+@3)", Cv, SM_BR_hww, SM_BR_hzz, SM_BR_hZg )')
        if self.universalCF:
            self.modelBuilder.factory_('expr::C5ql_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6)",\
             Cf, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hmm, SM_BR_hss)') 
        else:
            self.modelBuilder.factory_('expr::C5ql_Gscal_sumf("@0*@0 * (@1+@2+@3+@4) + @5*@5 * (@6+@7)",\
             Cq, SM_BR_hbb, SM_BR_hcc, SM_BR_htoptop, SM_BR_hss,\
             Cl, SM_BR_htt, SM_BR_hmm)') 
        self.modelBuilder.factory_('sum::C5ql_Gscal_tot(C5ql_Gscal_sumglu, C5ql_Gscal_sumg, C5ql_Gscal_sumv, C5ql_Gscal_sumf)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_("expr::C5ql_BRscal_hgg(\"@0*@0/@1\", Cg, C5ql_Gscal_tot)")
        self.modelBuilder.factory_("expr::C5ql_BRscal_hvv(\"@0*@0/@1\", Cv, C5ql_Gscal_tot)")
        if self.universalCF:
            self.modelBuilder.factory_("expr::C5ql_BRscal_hff(\"@0*@0/@1\", Cf, C5ql_Gscal_tot)")
        else:
            self.modelBuilder.factory_("expr::C5ql_BRscal_hbb(\"@0*@0/@1\", Cq, C5ql_Gscal_tot)")
            self.modelBuilder.factory_("expr::C5ql_BRscal_htt(\"@0*@0/@1\", Cl, C5ql_Gscal_tot)")
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "C5ql_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = "Cglu" if production in ["ggH"] else "Cv"
            if production in ['ttH']:
                XSscal = 'Cf' if self.universalCF else 'Cq'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hvv"
            if decay in ["hbb", "htt"]: BRscal = ("hff" if self.universalCF else decay)
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, C5ql_BRscal_%s)' % (name, XSscal, BRscal))
        return name

class C5udHiggs(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.floatMass = False
        self.universalCF = False
        self.fix = []
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po == "universalCF": universalCF = True
            if po.startswith("fix="): self.fix = po.replace("fix=","").split(",")
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
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("Cg[1,0,10]")
        self.modelBuilder.doVar("Cv[1,0,10]")
        self.modelBuilder.doVar("Cglu[1,0,10]")
        POI = "Cg,Cv,Cglu"
        if self.universalCF:
            self.modelBuilder.doVar("Cf[1,0,10]")
            POI += ",Cf"
        else:
            self.modelBuilder.doVar("Cu[1,0,10]")
            self.modelBuilder.doVar("Cd[1,0,10]")
            POI += ",Cu,Cd"
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            POI += ",MH"
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
        for F in self.fix:
           self.modelBuilder.out.var(F).setConstant(True)
           if F+"," in POI: POI = POI.replace(F+",", "")
           else:            POI = POI.replace(","+F, "")
        self.modelBuilder.doSet("POI",POI)
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::C5ud_Gscal_sumglu("@0*@0 * @1", Cglu, SM_BR_hgluglu)')
        self.modelBuilder.factory_('expr::C5ud_Gscal_sumg("@0*@0 * @1", Cg, SM_BR_hgg)')
        self.modelBuilder.factory_('expr::C5ud_Gscal_sumv("@0*@0 * (@1+@2+@3)", Cv, SM_BR_hww, SM_BR_hzz, SM_BR_hZg )')
        if self.universalCF:
            self.modelBuilder.factory_('expr::C5ud_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6)",\
             Cf, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hmm, SM_BR_hss)') 
        else:
            self.modelBuilder.factory_('expr::C5ud_Gscal_sumf("@0*@0 * (@1+@2+@3+@4) + @5*@5 * (@6+@7)",\
             Cd, SM_BR_hbb, SM_BR_hcc, SM_BR_htt, SM_BR_hmm,\
             Cu, SM_BR_htoptop, SM_BR_hss)') 
        self.modelBuilder.factory_('sum::C5ud_Gscal_tot(C5ud_Gscal_sumglu, C5ud_Gscal_sumg, C5ud_Gscal_sumv, C5ud_Gscal_sumf)')

        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM)^2 / (total/total_SM)^2 
        self.modelBuilder.factory_("expr::C5ud_BRscal_hgg(\"@0*@0/@1\", Cg, C5ud_Gscal_tot)")
        self.modelBuilder.factory_("expr::C5ud_BRscal_hvv(\"@0*@0/@1\", Cv, C5ud_Gscal_tot)")
        if self.universalCF:
            self.modelBuilder.factory_("expr::C5ud_BRscal_hff(\"@0*@0/@1\", Cf, C5ud_Gscal_tot)")
        else:
            self.modelBuilder.factory_("expr::C5ud_BRscal_hbb(\"@0*@0/@1\", Cd, C5ud_Gscal_tot)")
            self.modelBuilder.factory_("expr::C5ud_BRscal_htt(\"@0*@0/@1\", Cd, C5ud_Gscal_tot)")
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "C5ud_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = "Cglu" if production in ["ggH"] else "Cv"
            if production in ['ttH']:
                XSscal = 'Cf' if self.universalCF else 'Cu'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hvv"
            if decay in ["hbb", "htt"]: BRscal = ("hff" if self.universalCF else decay)
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, C5ud_BRscal_%s)' % (name, XSscal, BRscal))
        return name
