from HiggsAnalysis.CombinedLimit.PhysicsModel import *

### This base class implements signal yields by production and decay mode
### Specific models can be obtained redefining getHiggsSignalYieldScale
class TwoHypotesisHiggs(PhysicsModel):
    def __init__(self): 
        self.mHRange = []
        self.muAsPOI    = False
        self.muFloating = False
        self.poiMap  = []
        self.pois    = {}
        self.verbose = False
        self.altSignal  = "ALT"
    def setModelBuilder(self, modelBuilder):
        PhysicsModel.setModelBuilder(self, modelBuilder)
        self.modelBuilder.doModelBOnly = False
    def getYieldScale(self,bin,process):
        "Split in production and decay, and call getHiggsSignalYieldScale; return 1 for backgrounds "
        if not self.DC.isSignal[process]: return 1

        isAlt = (self.altSignal in process)

        if self.pois:
            target = "%(bin)s/%(process)s" % locals()
            scale = 1
            for p, l in self.poiMap:
                for el in l:
                    if re.match(el, target):
                        scale = p + self.sigNorms[isAlt]

            print "Will scale ", target, " by ", scale
            return scale;


        else:
            print "Process ", process, " will get norm ", self.sigNorms[isAlt]
            return self.sigNorms[isAlt]
    
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
            if po.startswith("verbose"):
                self.verbose = True
            if po.startswith("map="):
                self.muFloating = True
                (maplist,poi) = po.replace("map=","").split(":")
                maps = maplist.split(",")
                poiname = re.sub("\[.*","", poi)
                if poiname not in self.pois:
                    if self.verbose: print "Will create a var ",poiname," with factory ",poi
                    self.pois[poiname] = poi
                if self.verbose:  print "Mapping ",poiname," to ",maps," patterns"
                self.poiMap.append((poiname, maps))
                                                                                                            
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        self.modelBuilder.doVar("x[0,0,1]");
        poi = "x"

        if self.muFloating: 

            if self.pois:
                for pn,pf in self.pois.items():
                    self.modelBuilder.doVar(pf)
                    if self.muAsPOI:
                        print 'Treating %(pn)s as a POI' % locals()
                        poi += ','+pn
        
                    self.modelBuilder.factory_('expr::%(pn)s_times_not_x("@0*(1-@1)", %(pn)s, x)' % locals())
                    self.modelBuilder.factory_('expr::%(pn)s_times_x("@0*@1", %(pn)s, x)' % locals())
                self.sigNorms = { True:'_times_x', False:'_times_not_x' }
                    
            else:
                self.modelBuilder.doVar("r[1,0,4]");
                if self.muAsPOI:
                    print 'Treating r as a POI'
                    poi += ",r"

                self.modelBuilder.factory_("expr::r_times_not_x(\"@0*(1-@1)\", r, x)")
                self.modelBuilder.factory_("expr::r_times_x(\"@0*@1\", r, x)")
                self.sigNorms = { True:'r_times_x', False:'r_times_not_x' }

        else:
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

