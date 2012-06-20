from HiggsAnalysis.CombinedLimit.PhysicsModel import *

### This base class implements signal yields by production and decay mode
### Specific models can be obtained redefining getHiggsSignalYieldScale
class TwoHypotesisHiggs(PhysicsModel):
    def __init__(self): 
        self.mHRange = []
        self.muAsPOI    = False
        self.muFloating = False
        self.altSignal  = "ALT"
    def setModelBuilder(self, modelBuilder):
        PhysicsModel.setModelBuilder(self, modelBuilder)
        self.modelBuilder.doModelBOnly = False
    def getYieldScale(self,bin,process):
        "Split in production and decay, and call getHiggsSignalYieldScale; return 1 for backgrounds "
        if not self.DC.isSignal[process]: return 1
        #print "Process ",process," will get norm ",self.sigNorms[self.altSignal in process]
        return self.sigNorms[self.altSignal in process]
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po == "muAsPOI": 
                print "Will consider the signal strength as a parameter of interest"
                self.muAsPOI = True
                self.muFloating = True
            if po == "muFloating": 
                print "Will consider the signal strength as a floating parameter (as a parameter of interest if --PO muAsPOI is specified, as a nuisance otherwise)"
                self.muFloating = True
            if po.startswith("altSignal="): self.altSignal = po.split(",")[1]
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar("x[0,0,1]");
        poi = "x"
        if self.muFloating: 
            self.modelBuilder.doVar("r[1,0,4]");
            if self.muAsPOI: poi += ",r"
            self.modelBuilder.factory_("expr::r_times_not_x(\"@0*(1-@1)\", r, x)")
            self.modelBuilder.factory_("expr::r_times_x(\"@0*@1\", r, x)")
            #self.modelBuilder.factory_("expr::r_times_not_x(\"@0*(1-0.9999*@1)\", r, x)")
            #self.modelBuilder.factory_("expr::r_times_x(\"@0*(0.0001+0.9999*@1)\", r, x)")
            self.sigNorms = { True:'r_times_x', False:'r_times_not_x' }
        else:
            #self.modelBuilder.factory_("expr::not_x(\"(1.000 - 0.999*@0)\", x)")
            #self.modelBuilder.factory_("expr::yes_x(\"(0.001 + 0.999*@0)\", x)")
            #self.sigNorms = { True:'yes_x', False:'not_x' }
            self.modelBuilder.factory_("expr::not_x(\"(1-@0)\", x)")
            self.sigNorms = { True:'x', False:'not_x' }
        if self.modelBuilder.out.var("MH"):
            if len(self.mHRange):
                print 'MH will be left floating within', self.mHRange[0], 'and', self.mHRange[1]
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
                poi+=',MH'
            else:
                print 'MH will be assumed to be', self.options.mass
                self.modelBuilder.out.var("MH").removeRange()
                self.modelBuilder.out.var("MH").setVal(self.options.mass)
        else:
            if len(self.mHRange):
                print 'MH will be left floating within', self.mHRange[0], 'and', self.mHRange[1]
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1]))
                poi+=',MH'
            else:
                print 'MH (not there before) will be assumed to be', self.options.mass
                self.modelBuilder.doVar("MH[%g]" % self.options.mass)
        self.modelBuilder.doSet("POI",poi)

twoHypothesisHiggs = TwoHypotesisHiggs()

