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
        self.modelBuilder.doVar("CV[1,0,2]")
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
        self.modelBuilder.factory_('expr::CvCf_cgamma("\
        @0*@0*(1.524292518396496 + (0.005166702799572456 - 0.00003355715038472727*@2)*@2) + \
        @1*(@1*(0.07244520735564258 + (0.0008318872718720393 - 6.16997610275555e-6*@2)*@2) + \
        @0*(-0.5967377257521194 + (-0.005998590071444782 + 0.00003972712648748393*@2)*@2))\
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
        self.modelBuilder.doVar("Rw[1,0,10]")
        self.modelBuilder.factory_('prod::Rz(Rw, Rzw)')
            
        ## total witdhs, normalized to the SM one
        self.modelBuilder.factory_('expr::Rzw_Gscal_tot("@0*@1 + @2*@3 + (1.0-@1-@3)", \
                                   Rw, SM_BR_hww, Rz, SM_BR_hzz)')
        ## BRs, normalized to the SM ones: they scale as (partial/partial_SM) / (total/total_SM) 
        self.modelBuilder.factory_('expr::Rzw_BRscal_hww("@0/@1", Rw, Rzw_Gscal_tot)')
        self.modelBuilder.factory_('expr::Rzw_BRscal_hzz("@0/@1", Rz, Rzw_Gscal_tot)')
               
        
    def getHiggsSignalYieldScale(self,production,decay,energy):
        if decay not in ['hww', 'hzz']:
            return 0
        if production not in ['ggH']:
            return 1
        name = "Rzw_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name) == None: 
            self.modelBuilder.factory_('expr::%s("@0", Rzw_BRscal_%s)' % (name, decay))
        return name

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
        self.modelBuilder.factory_('prod::Cz(Cw, Czw)')
            
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

cZW  = CzwHiggs() 
rZW  = RzwHiggs()
cVcF = CvCfHiggs()
c5   = C5Higgs()

        
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
        


