
### Class that takes care of building a physics model by combining individual channels and processes together
### Things that it can do:
###   - define the parameters of interest (in the default implementation , "r")
###   - define other constant model parameters (e.g., "MH")
###   - yields a scaling factor for each pair of bin and process (by default, constant for background and linear in "r" for signal)
###   - possibly modifies the systematical uncertainties (does nothing by default)
class PhysicsModel:
    def __init__(self):
        pass
    def setModelBuilder(self, modelBuilder):
        "Connect to the ModelBuilder to get workspace, datacard and options. Should not be overloaded."
        self.modelBuilder = modelBuilder
        self.DC = modelBuilder.DC
        self.options = modelBuilder.options
    def setPhysicsOptions(self,physOptions):
        "Receive a list of strings with the physics options from command line"
        pass
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("r[0,20]");
        self.modelBuilder.doSet("POI","r")
        # --- Higgs Mass as other parameter ----
        if self.options.mass != 0:
            if self.modelBuilder.out.var("MH"):
              self.modelBuilder.out.var("MH").removeRange()
              self.modelBuilder.out.var("MH").setVal(self.options.mass)
            else:
              self.modelBuilder.doVar("MH[%g]" % self.options.mass); 
    def preProcessNuisances(self,nuisances):
        "receive the usual list of (name,nofloat,pdf,args,errline) to be edited"
        pass # do nothing by default
    def getYieldScale(self,bin,process):
        "Return the name of a RooAbsReal to scale this yield by or the two special values 1 and 0 (don't scale, and set to zero)"
        return "r" if self.DC.isSignal[process] else 1;


### This base class implements signal yields by production and decay mode
### Specific models can be obtained redefining getHiggsSignalYieldScale
class SMLikeHiggsModel(PhysicsModel):
    def getHiggsSignalYieldScale(self, production, decay, energy):
            raise RuntimeError, "Not implemented"
    def getYieldScale(self,bin,process):
        "Split in production and decay, and call getHiggsSignalYieldScale; return 1 for backgrounds "
        if not self.DC.isSignal[process]: return 1
        processSource = process
        decaySource   = self.options.fileName+":"+bin # by default, decay comes from the datacard name or bin label
        if "_" in process: (processSource, decaySource) = process.split("_")
        if processSource not in ["ggH", "qqH", "VH", "WH", "ZH", "ttH"]:
            raise RuntimeError, "Validation Error: signal process %s not among the allowed ones." % processSource

        foundDecay = None
        for D in [ "hww", "hzz", "hgg", "htt", "hbb" ]:
            if D in decaySource:
                if foundDecay: raise RuntimeError, "Validation Error: decay string %s contains multiple known decay names" % decaySource
                foundDecay = D
        if not foundDecay: raise RuntimeError, "Validation Error: decay string %s does not contain any known decay name" % decaySource

        foundEnergy = None
        for D in [ '7TeV', '8TeV', '14TeV' ]:
            if D in decaySource:
                if foundEnergy: raise RuntimeError, "Validation Error: decay string %s contains multiple known energies" % decaySource
                foundEnergy = D
        if not foundEnergy:
            foundEnergy = '7TeV' ## To ensure backward compatibility
            print "Warning: decay string %s does not contain any known energy, assuming %s" % (decaySource, foundEnergy)

        return self.getHiggsSignalYieldScale(processSource, foundDecay, foundEnergy)

class StrictSMLikeHiggsModel(SMLikeHiggsModel):
    "Doesn't do anything more, but validates that the signal process names are correct"
    def getHiggsSignalYieldScale(self,production,decay, energy):
            return "r"

class FloatingHiggsMass(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs mass to float"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.mHRange = ['115','135'] # default
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrama for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI out of signal strength and MH"""
        # --- Signal Strength as only POI --- 
        self.modelBuilder.doVar("r[1,0,20]")
        if self.modelBuilder.out.var("MH"):
            self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
            self.modelBuilder.out.var("MH").setConstant(False)
        else:
            self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
        self.modelBuilder.doSet("POI",'r,MH')
    def getHiggsSignalYieldScale(self,production,decay, energy):
            return "r"


class FloatingXSHiggs(SMLikeHiggsModel):
    "Float independently ggH and qqH cross sections"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.modes = [ "ggH", "qqH", "VH", "ttH" ]
        self.mHRange = []
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("modes="): self.modes = po.replace("modes=","").split(",")
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrama for Higgs mass range defined with inverterd order. Second must be larger the first"
    def doParametersOfInterest(self):
        """Create POI and other parameters, and define the POI set."""
        # --- Signal Strength as only POI --- 
        if "ggH" in self.modes: self.modelBuilder.doVar("r_ggH[1,0,5]");
        if "qqH" in self.modes: self.modelBuilder.doVar("r_qqH[1,0,20]");
        if "VH"  in self.modes: self.modelBuilder.doVar("r_VH[1,0,20]");
        if "ttH" in self.modes: self.modelBuilder.doVar("r_ttH[1,0,20]");
        poi = ",".join(["r_"+m for m in self.modes])
        # --- Higgs Mass as other parameter ----
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
    def getHiggsSignalYieldScale(self,production,decay, energy):
        if production == "ggH": return ("r_ggH" if "ggH" in self.modes else 1)
        if production == "qqH": return ("r_qqH" if "qqH" in self.modes else 1)
        if production == "ttH": return ("r_ttH" if "ttH" in self.modes else 1)
        if production in [ "WH", "ZH", "VH" ]: return ("r_VH" if "VH" in self.modes else 1)
        raise RuntimeError, "Unknown production mode '%s'" % production

defaultModel = PhysicsModel()
strictSMLikeHiggs = StrictSMLikeHiggsModel()
floatingXSHiggs = FloatingXSHiggs()
floatingHiggsMass = FloatingHiggsMass()
