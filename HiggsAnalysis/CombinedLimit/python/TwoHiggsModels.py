from HiggsAnalysis.CombinedLimit.PhysicsModel import *

### Scale the signal higgs at MH by 'r'
### Scale the backgroud higgs_SM at MH_SM by r_SM
### r_SM can be fixed or floating (as nuisance or POI)
class TwoHiggsBase(PhysicsModel):
    def __init__(self): 
        self.mHSM      = 125.8
        self.mHRange = []
        self.mHSMRange = []
        self.altSignal  = "_SM"
        self.modes = [ "ggH", "qqH", "ttH", "WH", "ZH", "VH" ]
        self.mHAsPOI   = False
        self.mHSMAsPOI = False
    def getHiggsYieldScaleSM(self,production,decay, energy):
        return 1   ## to be implemented in subclasses
    def getHiggsYieldScale(self,production,decay, energy):
        return "r" ## to be implemented in subclasses
    def getYieldScale(self,bin,process):
        if self.DC.isSignal[process] and process in self.modes: 
            (production,decay, energy) = getHiggsProdDecMode(bin,process,self.options)
            return self.getHiggsYieldScale(production,decay, energy)
        if process.endswith(self.altSignal) and process.replace(self.altSignal,"") in self.modes: 
            (production,decay, energy) = getHiggsProdDecMode(bin,process.replace("_SM",""),self.options)
            return self.getHiggsYieldScaleSM(production,decay, energy)
        return 1;
    def setPhysicsOptionsBase(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassSM="):
                self.mHSM = float((po.replace("higgsMassSM=","").split(","))[0]) # without index, this will try to case list as a float !
            if po == "mHAsPOI": 
                print "Will consider the mass of the second Higgs as a parameter of interest"
                self.mHAsPOI = True
            if po == "mHSMAsPOI": 
                print "Will consider the mass of the SM Higgs as a parameter of interest"
                self.mHSMAsPOI = True
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first"
            if po.startswith("higgsMassRangeSM="):
                self.mHSMRange = po.replace("higgsMassRangeSM=","").split(",")
                if len(self.mHSMRange) != 2:
                    raise RuntimeError, "SM Higgs mass range definition requires two extrema"
                elif float(self.mHSMRange[0]) >= float(self.mHSMRange[1]):
                    raise RuntimeError, "Extrema for SM Higgs mass range defined with inverterd order. Second must be larger the first"
    def doMasses(self): 
        """Create mass variables, return a postfix to the POIs if needed"""
        poi = ""
        ## Searched-for higgs
        if self.modelBuilder.out.var("MH"):
            if len(self.mHRange):
                print 'MH will be left floating within', self.mHRange[0], 'and', self.mHRange[1]
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
                if self.mHAsPOI: poi+=',MH'
            else:
                print 'MH will be assumed to be', self.options.mass
                self.modelBuilder.out.var("MH").removeRange()
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
        else:
            if len(self.mHRange):
                print 'MH will be left floating within', self.mHRange[0], 'and', self.mHRange[1]
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1]))
                if self.mHAsPOI: poi+=',MH'
            else:
                print 'MH (not there before) will be assumed to be', self.options.mass
                self.modelBuilder.doVar("MH[%g]" % self.options.mass)
        ## Already-found higgs
        if self.modelBuilder.out.var("MH_SM"):
            if len(self.mHSMRange):
                print 'MH_SM will be left floating within', self.mHSMRange[0], 'and', self.mHSMRange[1]
                self.modelBuilder.out.var("MH_SM").setRange(float(self.mHSMRange[0]),float(self.mHSMRange[1]))
                self.modelBuilder.out.var("MH_SM").setConstant(False)
                if self.mHSMAsPOI: poi+=',MH_SM'
            else:
                print 'MH_SM will be assumed to be', self.mHSM
                self.modelBuilder.out.var("MH_SM").removeRange()
                self.modelBuilder.out.var("MH_SM").setVal(self.mHSM)
        else:
            if len(self.mHSMRange):
                print 'MH_SM will be left floating within', self.mHSMRange[0], 'and', self.mHSMRange[1]
                self.modelBuilder.doVar("MH_SM[%s,%s]" % (self.mHSMRange[0],self.mHSMRange[1]))
                if self.mHSMAsPOI: poi+=',MH_SM'
            else:
                print 'MH_SM (not there before) will be assumed to be', self.mHSM
                self.modelBuilder.doVar("MH_SM[%g]" % self.mHSM)
        return poi

class JustOneHiggs(TwoHiggsBase):
    ## only put in the second Higgs, as a cross-check of the 1-higgs models
    def __init__(self): 
        TwoHiggsBase.__init__(self)
        self.muSMAsPOI    = False
        self.muSMFloating = False
    def getHiggsYieldScaleSM(self,production,decay, energy):
        return 0
    def getHiggsYieldScale(self,production,decay, energy):
        return "r" 
    def setPhysicsOptions(self,physOptions):
        self.setPhysicsOptionsBase(physOptions)
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # take care of the searched-for Higgs yield
        self.modelBuilder.doVar("r[1,0,4]");
        poi = "r"
        # take care of masses
        poi += self.doMasses()
        # done
        self.modelBuilder.doSet("POI",poi)

class TwoHiggsUnconstrained(TwoHiggsBase):
    def __init__(self): 
        TwoHiggsBase.__init__(self)
        self.muSMAsPOI    = False
        self.muSMFloating = False
    def getHiggsYieldScaleSM(self,production,decay, energy):
        return "r_SM" 
    def getHiggsYieldScale(self,production,decay, energy):
        return "r" 
    def setPhysicsOptions(self,physOptions):
        self.setPhysicsOptionsBase(physOptions)
        for po in physOptions:
            if po == "muSMAsPOI": 
                print "Will consider the signal strength of the SM Higgs as a parameter of interest"
                self.muSMAsPOI = True
                self.muSMFloating = True
            if po == "muSMFloating": 
                print "Will consider the signal strength of the SM Higgs as a floating parameter (as a parameter of interest if --PO muAsPOI is specified, as a nuisance otherwise)"
                self.muSMFloating = True
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # take care of the searched-for Higgs yield
        self.modelBuilder.doVar("r[1,0,4]");
        poi = "r"
        # take care of the SM Higgs yield
        if self.muSMFloating: self.modelBuilder.doVar("r_SM[1,0,4]"); 
        else:                 self.modelBuilder.doVar("r_SM[1]");
        if self.muSMAsPOI: poi += ",r_SM"
        # take care of masses
        poi += self.doMasses()
        # done
        self.modelBuilder.doSet("POI",poi)

class SingletMixing(TwoHiggsBase):
    """ 3 Possibilities for the signal yields:
                           MH Higgs        SM Higgs
      - default:           r               r_SM := 1 - r
      - with BSM:          r*(1-BR_BSM)    r_SM := 1 - r
      - BSM & Visible mu:  r               r_SM := 1 - r/(1-BR_BSM) 
      note that in the third case there's no way to enforce at model level that r/(1-BR_BSM) <= 1
    """
    def __init__(self): 
        TwoHiggsBase.__init__(self)
        self.withBSM = False
        self.useVisibleMu = False
    def getHiggsYieldScaleSM(self,production,decay, energy):
        return "one_minus_r"
    def getHiggsYieldScale(self,production,decay, energy):
        return "r_times_BR" if (self.withBSM and not self.useVisibleMu) else "r"
    def setPhysicsOptions(self,physOptions):
        self.setPhysicsOptionsBase(physOptions)
        for po in physOptions:
            if po == "BSMDecays": 
                self.withBSM = True
            if po == "UseVisibleMu": 
                self.useVisibleMu = True
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # take care of the searched-for Higgs yield
        self.modelBuilder.doVar("r[1,0,1]");
        poi = "r"
        # and of the SM one
        if self.withBSM and self.useVisibleMu:
            self.modelBuilder.factory_("expr::one_minus_r(\"max(0,1-@0/(1-@1))\", r, BR_BSM[0,0,1])");
        else:
            self.modelBuilder.factory_("expr::one_minus_r(\"(1-@0)\", r)");
        # if BSM decays are allowed
        if self.withBSM:
            if not self.useVisibleMu:
                self.modelBuilder.factory_("expr::r_times_BR(\"@0*(1-@1)\", r, BR_BSM[0,0,1])");
            poi += ",BR_BSM"
        # take care of masses
        poi += self.doMasses()
        # done
        self.modelBuilder.doSet("POI",poi)

class SingletMixingForExclusion(TwoHiggsBase):
    """ The prediction for this model is mu + mu' == 1 (or <= 1)
        So we just go probing mu+mu' """
    def __init__(self): 
        TwoHiggsBase.__init__(self)
        self.withBSM = False
    def getHiggsYieldScaleSM(self,production,decay, energy):
        return "r_times_x"
    def getHiggsYieldScale(self,production,decay, energy):
        return "r_times_BR_times_not_x" if self.withBSM else "r_times_not_x"
    def setPhysicsOptions(self,physOptions):
        self.setPhysicsOptionsBase(physOptions)
        for po in physOptions:
            if po == "BSMDecays": 
                self.withBSM = True
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # take care of the searched-for Higgs yield
        self.modelBuilder.doVar("r[1,0,10]");
        self.modelBuilder.doVar("x[1,0,1]");
        poi = "r,x"
        # take care of the searched-for Higgs yield
        if self.withBSM:
            self.modelBuilder.factory_("expr::r_times_BR_times_not_x(\"@0*max(0,1-@1)*(1-@2)\", r, x, BR_BSM[0,0,1])");
            poi += ",BR_BSM"
        else:
            self.modelBuilder.factory_("expr::r_times_not_x(\"@0*max(0,1-@1)\", r, x)");
        # and of the SM one
        self.modelBuilder.factory_("expr::r_times_x(\"@0*@1\", r, x)");
        # take care of masses
        poi += self.doMasses()
        # done
        self.modelBuilder.doSet("POI",poi)

class TwoHiggsCvCf(TwoHiggsBase):
    def __init__(self): 
        TwoHiggsBase.__init__(self)
        self.cVRange = ['0','2']
        self.cFRange = ['-2','2']

    def setPhysicsOptions(self,physOptions):
        self.setPhysicsOptionsBase(physOptions)
        for po in physOptions:
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
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar("CV[1,%s,%s]" % (self.cVRange[0], self.cVRange[1]))
        self.modelBuilder.doVar("CF[1,%s,%s]" % (self.cFRange[0], self.cFRange[1]))

        self.modelBuilder.doSet("POI",'CV,CF')

        self.modelBuilder.factory_('expr::CV_SM("sqrt(1-@0*@0)", CV)') 
        self.modelBuilder.factory_('expr::CF_SM("(1-@0*@1)/@2", CV, CF, CV_SM)') 
        
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
        self.modelBuilder.factory_('expr::2HCvCf_Gscal_sumf("@0*@0 * (@1+@2+@3+@4+@5+@6+@7)", CF, SM_BR_hbb, SM_BR_htt, SM_BR_hcc, SM_BR_htoptop, SM_BR_hgluglu, SM_BR_hmm, SM_BR_hss)') 
        self.modelBuilder.factory_('expr::2HCvCf_Gscal_sumv("@0*@0 * (@1+@2)", CV, SM_BR_hww, SM_BR_hzz)') 
        self.modelBuilder.factory_('expr::2HCvCf_Gscal_gg("@0 * @1", Scaling_hgg, SM_BR_hgg)') 
        self.modelBuilder.factory_('expr::2HCvCf_Gscal_Zg("@0 * @1", Scaling_hZg, SM_BR_hZg)') 
        self.modelBuilder.factory_('sum::2HCvCf_Gscal_tot(2HCvCf_Gscal_sumf, 2HCvCf_Gscal_sumv, 2HCvCf_Gscal_gg, 2HCvCf_Gscal_Zg)')
        ## BRs, normalized to the SM ones: they scale as (coupling/coupling_SM)^2 / (totWidth/totWidthSM)^2 
        self.modelBuilder.factory_('expr::2HCvCf_BRscal_hgg("@0/@1", Scaling_hgg, 2HCvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::2HCvCf_BRscal_hZg("@0/@1", Scaling_hZg, 2HCvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::2HCvCf_BRscal_hff("@0*@0/@1", CF, 2HCvCf_Gscal_tot)')
        self.modelBuilder.factory_('expr::2HCvCf_BRscal_hvv("@0*@0/@1", CV, 2HCvCf_Gscal_tot)')

        

    def getHiggsYieldScaleSM(self,production,decay, energy):
        name = "2HCvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name):
            return name
        
        XSscal = self.productionScaling[production]
        BRscal = self.decayScaling[decay]
        self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, 2HCvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name

    def getHiggsYieldScale(self,production,decay, energy):
        name = "2HCvCf_XSBRscal_%s_%s" % (production,decay)
        if self.modelBuilder.out.function(name):
            return name
        
        XSscal = self.productionScaling[production]
        BRscal = self.decayScaling[decay]
        self.modelBuilder.factory_('expr::%s("@0*@0 * @1", %s, 2HCvCf_BRscal_%s)' % (name, XSscal, BRscal))
        return name


twoHiggsUnconstrained = TwoHiggsUnconstrained()
justOneHiggs = JustOneHiggs()
singletMixing = SingletMixing()
singletMixingForExclusion = SingletMixingForExclusion()

twoHiggsCvCf = TwoHiggsCvCf()
