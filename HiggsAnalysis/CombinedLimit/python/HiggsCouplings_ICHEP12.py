from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder
import ROOT, os

class CvCfHiggs(SMLikeHiggsModel):
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
        self.setup()

    def setup(self):
        ## Coefficient for couplings to photons
        #      arXiv 1202.3144v2, below eq. 2.6:  2/9*cF - 1.04*cV, and then normalize to SM 
        #      FIXME: this should be replaced with the proper MH dependency
        #self.modelBuilder.factory_("expr::CvCf_cgamma(\"-0.271*@0+1.27*@1\",CF,CV)")
        #########################################################################
        ## Coefficient for coupling to di-photons
        #      Based on Eq 1--4 of Nuclear Physics B 453 (1995)17-82
        #      ignoring b quark contributions
        # Taylor series around MH=125 including terms up to O(MH-125)^2 in Horner polynomial form
        CF = self.modelBuilder.out.function('CF')
        CF.setVal(1.0)
        self.modelBuilder.factory_('expr::CvCf_cgammaSq("\
        @0*@0*(1.524292518396496 + (0.005166702799572456 - 0.00003355715038472727*@2)*@2) + \
        @1*(@1*(0.07244520735564258 + (0.0008318872718720393 - 6.16997610275555e-6*@2)*@2) + \
        @0*(-0.5967377257521194 + (-0.005998590071444782 + 0.00003972712648748393*@2)*@2))\
        ",CV,CF,MH)')
        ## partial witdhs, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_("expr::CvCf_Gscal_sumf(\"@0*@0 * (@1+@2+@3+@4+@5+@6+@7)\", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)") 
        self.modelBuilder.factory_("expr::CvCf_Gscal_sumv(\"@0*@0 * (@1+@2+@3)\", CV, SM_BR_hww, SM_BR_hzz, SM_BR_hZg)") 
        self.modelBuilder.factory_("expr::CvCf_Gscal_gg(\"@0 * @1\", CvCf_cgammaSq, SM_BR_hgg)") 
        self.modelBuilder.factory_( "sum::CvCf_Gscal_tot(CvCf_Gscal_sumf, CvCf_Gscal_sumv, CvCf_Gscal_gg)")
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_("expr::CvCf_BRscal_hgg(\"@0/@1\", CvCf_cgammaSq, CvCf_Gscal_tot)")
        self.modelBuilder.factory_("expr::CvCf_BRscal_hf(\"@0*@0/@1\", CF, CvCf_Gscal_tot)")
        self.modelBuilder.factory_("expr::CvCf_BRscal_hv(\"@0*@0/@1\", CV, CvCf_Gscal_tot)")
        #self.modelBuilder.out.Print()
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "CvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = 'CF' if production in ["ggH","ttH"] else 'CV'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = "hf"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name

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


class RzwHiggs(SMLikeHiggsModel):
    "scale WW by mu and ZZ by cZW^2 * mu"
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
        self.modelBuilder.doVar("Rzw[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Rzw,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Rzw')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Rhww[1,0,10]")
        self.modelBuilder.factory_('expr::Rhzz("@0*@1", Rhww, Rzw)')               
        
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        else:
            return 'R%s' % decay

class RwzHiggs(SMLikeHiggsModel):
    "scale WW by mu and ZZ by cZW^2 * mu"
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
        self.modelBuilder.doVar("Rwz[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Rwz,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Rwz')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()
        
    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Rhzz[1,0,10]")
        self.modelBuilder.factory_('expr::Rhww("@0*@1", Rhzz, Rwz)')
               
        
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        else:
            return 'R%s' % decay



class CzwHiggs(SMLikeHiggsModel):
    "Scale w and z and touch nothing else"
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
        self.modelBuilder.doVar("Czw[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Czw,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Czw')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Cw[1,0,10]")
        self.modelBuilder.factory_('expr::Cz("@0*@1",Cw, Czw)')
            
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::Czw_Gscal_tot("@0*@1 + @2*@3 + (1.0-@1-@3)", \
                                   Cw, SM_BR_hww, Cz, SM_BR_hzz)')
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::Czw_BRscal_hww("@0/@1", Cw, Czw_Gscal_tot)')
        self.modelBuilder.factory_('expr::Czw_BRscal_hzz("@0/@1", Cz, Czw_Gscal_tot)')
        
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        for e in ['7TeV', '8TeV']:
            print 'build for %s'%e
            self.SMH.textToSpline(   'RqqH_%s'%e, os.path.join(datadir, 'couplings/R_VBF_%s.txt'%e), ycol=1 );
            self.modelBuilder.factory_('expr::Czw_XSscal_qqH_%s("(@0 + @1*@2) / (1.0 + @2) ", Cw, Cz, RqqH_%s)'%(e,e))
            self.modelBuilder.factory_('expr::Czw_XSscal_WH_%s("@0", Cw)'%e)
            self.modelBuilder.factory_('expr::Czw_XSscal_ZH_%s("@0", Cz)'%e)
            self.SMH.makeXS('WH',e)
            self.SMH.makeXS('ZH',e)
            self.modelBuilder.factory_('expr::Czw_XSscal_VH_%s("(@0*@1 + @2*@3) / (@1 + @3) ", Cw, SM_XS_WH_%s, Cz, SM_XS_ZH_%s)'%(e,e,e))

    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        
        name = "Czw_XSBRscal_%s_%s_%s" % (production,decay,energy)
        if self.modelBuilder.out.function(name) == None: 
            if production in ["ggH","ttH"]:
                self.modelBuilder.factory_('expr::%s("@0", Czw_BRscal_%s)' % (name, decay))
            else:
                self.modelBuilder.factory_('expr::%s("@0 * @1", Czw_XSscal_%s_%s, Czw_BRscal_%s)' % (name, production, energy, decay))
        return name

class CwzHiggs(SMLikeHiggsModel):
    "Scale w and z and touch nothing else"
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
        self.modelBuilder.doVar("Cwz[1,0,10]")
        if self.floatMass:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
            self.modelBuilder.doSet("POI",'Cwz,MH')
        else:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
                self.modelBuilder.out.var("MH").setConstant(True)
            else:
                self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
            self.modelBuilder.doSet("POI",'Cwz')
        self.SMH = SMHiggsBuilder(self.modelBuilder)
        self.setup()

    def setup(self):
        for d in [ "hww", "hzz" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.doVar("Cz[1,0,10]")
        self.modelBuilder.factory_('expr::Cw("@0*@1",Cz, Cwz)')
            
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::Cwz_Gscal_tot("@0*@1 + @2*@3 + (1.0-@1-@3)", \
                                   Cw, SM_BR_hww, Cz, SM_BR_hzz)')
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::Cwz_BRscal_hww("@0/@1", Cw, Cwz_Gscal_tot)')
        self.modelBuilder.factory_('expr::Cwz_BRscal_hzz("@0/@1", Cz, Cwz_Gscal_tot)')
        
        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
        for e in ['7TeV', '8TeV']:
            print 'build for %s'%e
            self.SMH.textToSpline(   'RqqH_%s'%e, os.path.join(datadir, 'couplings/R_VBF_%s.txt'%e), ycol=1 );
            self.modelBuilder.factory_('expr::Cwz_XSscal_qqH_%s("(@0 + @1*@2) / (1.0 + @2) ", Cw, Cz, RqqH_%s)'%(e,e))
            self.modelBuilder.factory_('expr::Cwz_XSscal_WH_%s("@0", Cw)'%e)
            self.modelBuilder.factory_('expr::Cwz_XSscal_ZH_%s("@0", Cz)'%e)
            self.SMH.makeXS('WH',e)
            self.SMH.makeXS('ZH',e)
            self.modelBuilder.factory_('expr::Cwz_XSscal_VH_%s("(@0*@1 + @2*@3) / (@1 + @3) ", Cw, SM_XS_WH_%s, Cz, SM_XS_ZH_%s)'%(e,e,e))

    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        
        name = "Cwz_XSBRscal_%s_%s_%s" % (production,decay,energy)
        if self.modelBuilder.out.function(name) == None: 
            if production in ["ggH","ttH"]:
                self.modelBuilder.factory_('expr::%s("@0", Cwz_BRscal_%s)' % (name, decay))
            else:
                self.modelBuilder.factory_('expr::%s("@0 * @1", Cwz_XSscal_%s_%s, Cwz_BRscal_%s)' % (name, production, energy, decay))
        return name

cWZ  = CwzHiggs() 
cZW  = CzwHiggs() 
rZW  = RzwHiggs()
rWZ  = RwzHiggs()
cVcF = CvCfHiggs()
c5ql = C5qlHiggs()
c5ud = C5udHiggs()
        
#        ## Add some common ingredients
#        self.modelBuilder.doVar("mt[172.5]")
#        datadir = os.environ['CMSSW_BASE']+'/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg'
#        self.SMH.textToSpline('alpha_s', os.path.join(datadir, 'running_constants.txt'), ycol=1 );
#        self.SMH.textToSpline(     'mb', os.path.join(datadir, 'running_constants.txt'), ycol=2 );
#                
#        ## Add some more ingredients tailored to this model's needs       
#        mH = self.modelBuilder.out.var('MH')
#        CF = self.modelBuilder.out.var('CF')
#        CV = self.modelBuilder.out.var('CV')
#        mt = self.modelBuilder.out.var('mt')
#        mb = self.modelBuilder.out.function('mb')
#        alpha_s = self.modelBuilder.out.function('alpha_s')
#                
#        CV.setVal(1.05957)
#        CF.setVal(0.715628) 
#        
#        Rgamgam = ROOT.RooScaleHGamGamLOSM('Rgamgam', 'LO SM Hgamgam scaling', mH, CF, CF, CV, mt, mb)
#        self.modelBuilder.out._import(Rgamgam)
#        Rzgam = ROOT.RooScaleHZGamLOSM('Rzgam', 'LO SM HZgam scaling', mH, CF, CF, CV, mt, mb)
#        self.modelBuilder.out._import(Rzgam)
#        Rgluglu = ROOT.RooScaleHGluGluLOSM('Rgluglu', 'LO SM Hgluglu scaling', mH, CF, CF, mt, mb, alpha_s)
#        self.modelBuilder.out._import(Rgluglu)
#        
#        mH.Print()
#        mt.Print()
#        CV.Print()
#        CF.Print()
#        self.modelBuilder.out.Print()
        


