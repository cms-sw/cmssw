
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
        self.modelBuilder = modelBuilder
        self.DC = modelBuilder.DC
        self.options = modelBuilder.options
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


class StrictSMLikeHiggsModel(PhysicsModel):
    def getYieldScale(self,bin,process):
        "Doesn't do anything more, but validates that the signal process names are correct"
        if self.DC.isSignal[process]:
            processSource = process
            decaySource   = self.options.fileName+":"+bin # by default, decay comes from the datacard name or bin label
            if "_" in process: (processSource, decaySource) = process.split("_")
            if processSource not in ["ggH", "qqH", "VH", "WH", "ZH", "ttH"]:
                raise RuntimeError, "Validation Error: signal process %s not among the allowed ones." % processSource
            foundDecay = False
            for D in [ "hww", "hzz", "hgg", "htt", "hbb" ]:
                if D in decaySource:
                    if foundDecay: raise RuntimeError, "Validation Error: decay string %s contains multiple known decay names" % decaySource
                    foundDecay = True
            if not foundDecay: raise RuntimeError, "Validation Error: decay string %s does not contain any known decay name" % decaySource
        return "r" if self.DC.isSignal[process] else 1;

defaultModel = PhysicsModel()
strictSMLikeHiggs = StrictSMLikeHiggsModel()
