from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.SMHiggsBuilder import SMHiggsBuilder

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
        self.modelBuilder.doVar("CV[1,-5,5]")
        self.modelBuilder.doVar("CF[1,-5,5]")
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
    def doDebugDump(self):
        self.modelBuilder.out.var("MH").setConstant(False)
        self.modelBuilder.out.var("MH").removeRange()
        MHvals = [ 110 + (600.-110.)*i/4900. for i in xrange(4900+1) ]
        for p in [ "ggH", "qqH", "WH", "ZH" ]: 
            self.SMH.makeXS(p)
            self.SMH.dump("SM_XS_"+p, "MH", MHvals, "dump.XS_"+p+".txt")
        for p in [ "htt", "hbb", "hww", "hzz", "hgg", "hgluglu", "htoptop" ]:
            self.SMH.makeBR(p)
            self.SMH.dump("SM_BR_"+p, "MH", MHvals, "dump.BR_"+p+".txt")
        self.SMH.makeTotalWidth()
        self.SMH.dump("SM_GammaTot", "MH", MHvals, "dump.GammaTot.txt")
    def setup(self):
        ## Coefficient for couplings to photons
        #      arXiv 1202.3144v2, below eq. 2.6:  2/9*cF - 1.04*cV, and then normalize to SM 
        #      FIXME: this should be replaced with the proper MH dependency
        #self.modelBuilder.factory_("expr::CvCf_cgamma(\"-0.271*@0+1.27*@1\",CF,CV)")
        #
        # Taylor series around MH=125 to (MH-125)^2 in Horner polynomial form
        self.modelBuilder.factory_('expr::CvCf_cgamma("\
        @0*(1.2259236555204187 + (0.00216740776385032 - 0.000013693587140986294*@2)*@2) +\
        @1*(-0.22592365552041888 + (-0.002167407763850317 + 0.000013693587140986278*@2)*@2)\
        ",CV,CF,MH)')
        ## partial witdhs, normalized to the SM one, for decays scaling with F, V and total
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        self.modelBuilder.factory_("expr::CvCf_Gscal_sumf(\"@0*@0 * (@1+@2+@3+@4+@5+@6+@7)\", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)") 
        self.modelBuilder.factory_("expr::CvCf_Gscal_sumv(\"@0*@0 * (@1+@2+@3)\", CV, SM_BR_hww, SM_BR_hzz, SM_BR_hZg)") 
        self.modelBuilder.factory_("expr::CvCf_Gscal_gg(\"@0*@0 * @1\", CvCf_cgamma, SM_BR_hgg)") 
        self.modelBuilder.factory_( "sum::CvCf_Gscal_tot(CvCf_Gscal_sumf, CvCf_Gscal_sumv, CvCf_Gscal_gg)")
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_("expr::CvCf_BRscal_hgg(\"@0*@0/@1\", CvCf_cgamma, CvCf_Gscal_tot)")
        self.modelBuilder.factory_("expr::CvCf_BRscal_hf(\"@0*@0/@1\", CF, CvCf_Gscal_tot)")
        self.modelBuilder.factory_("expr::CvCf_BRscal_hv(\"@0*@0/@1\", CV, CvCf_Gscal_tot)")
        ## XS*BR scales
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "CvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = 'CF' if production in ["ggH","ttH"] else 'CV'
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = "hf"
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, CvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name

class C5Higgs(SMLikeHiggsModel):
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
        self.modelBuilder.doVar("Cgg[1,0,10]")
        self.modelBuilder.doVar("Cvv[1,0,10]")
        self.modelBuilder.doVar("Cgluglu[1,0,10]")
        POI = "Cgg,Cvv,Cgluglu"
        if self.universalCF:
            self.modelBuilder.doVar("Cff[1,0,10]")
            POI += ",Cff"
        else:
            self.modelBuilder.doVar("Cbb[1,0,10]")
            self.modelBuilder.doVar("Ctt[1,0,10]")
            POI += ",Cbb,Ctt"
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
        #self.doDebugDump()
        self.setup()
    def setup(self):
        for d in [ "htt", "hbb", "hcc", "hww", "hzz", "hgluglu", "htoptop", "hgg", "hZg", "hmm", "hss" ]:
            self.SMH.makeBR(d)
        ## total witdhs, normalized to the SM one
        if self.universalCF:
            self.modelBuilder.factory_("expr::C5_Gscal_tot(\"@0*@1 + @2*(@3+@4+@5+@9+@10+@11) + @6*(@7+@8)\","+
                                       " Cgluglu, SM_BR_hgluglu, Cff, SM_BR_hbb, SM_BR_hcc, SM_BR_htt,"+
                                       " Cvv, SM_BR_hww, SM_BR_hzz,   SM_BR_hss, SM_BR_hmm, SM_BR_htoptop)")
        else:
            self.modelBuilder.factory_("expr::C5_Gscal_tot(\"@0*@1 + @2*@3 + @4 + @5*@6 + @7*(@8+@9)\","+
                                       " Cgluglu, SM_BR_hgluglu, Cbb, SM_BR_hbb, SM_BR_hcc, Ctt, SM_BR_htt,"+
                                       " Cvv, SM_BR_hww, SM_BR_hzz)")
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_("expr::C5_BRscal_hgg(\"@0/@1\", Cgg, C5_Gscal_tot)")
        self.modelBuilder.factory_("expr::C5_BRscal_hv(\"@0/@1\",  Cvv, C5_Gscal_tot)")
        if self.universalCF:
            self.modelBuilder.factory_("expr::C5_BRscal_hf(\"@0/@1\", Cff, C5_Gscal_tot)")
        else:
            self.modelBuilder.factory_("expr::C5_BRscal_hbb(\"@0/@1\", Cbb, C5_Gscal_tot)")
            self.modelBuilder.factory_("expr::C5_BRscal_htt(\"@0/@1\", Ctt, C5_Gscal_tot)")
    def getHiggsSignalYieldScale(self,production,decay,energy):
        name = "C5_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            XSscal = "Cgluglu" if production in ["ggH"] else "Cvv"
            BRscal = "hgg"
            if decay in ["hww", "hzz"]: BRscal = "hv"
            if decay in ["hbb", "htt"]: BRscal = ("hf" if self.universalCF else decay)
            self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, C5_BRscal_%s)' % (name, XSscal, BRscal))
        return name

cVcF = CvCfHiggs()
c5   = C5Higgs()
