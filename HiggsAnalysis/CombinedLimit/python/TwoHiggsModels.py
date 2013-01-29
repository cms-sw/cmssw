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
                self.mHSM = float(po.replace("higgsMassSM=","").split(","))
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

twoHiggsUnconstrained = TwoHiggsUnconstrained()
justOneHiggs = JustOneHiggs()
singletMixing = SingletMixing()
singletMixingForExclusion = SingletMixingForExclusion()

