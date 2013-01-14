from HiggsAnalysis.CombinedLimit.PhysicsModel import *

class FloatingHiggsWidth(SMLikeHiggsModel):
    "assume the SM coupling but let the Higgs Total Width to float, i.e. allowing invisible decay"
    def __init__(self):
        SMLikeHiggsModel.__init__(self) # not using 'super(x,self).__init__' since I don't understand it
        self.widthRange = ['0.001','5000'] # default
        self.rMode   = 'poi'
	self.mHRange = ['115','135']
	self.floatMass = False
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsWidthRange="):
                self.widthRange = po.replace("higgsWidthRange=","").split(",")
                print 'The Higgs Width range:', self.widthRange
                if len(self.widthRange) != 2:
                    raise RuntimeError, "Higgs Width range definition requires two extrema"
                elif float(self.widthRange[0]) >= float(self.widthRange[1]):
                    raise RuntimeError, "Extrama for Higgs Width range defined with inverterd order. Second must be larger the first"
            if po.startswith("signalStrengthMode="):
                self.rMode = po.replace("signalStrengthMode=","")
            if po.startswith("higgsMassRange="):
                self.floatMass = True
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema."
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first."
 
    def doParametersOfInterest(self):
        """Create POI out of signal strength and Width"""
        # --- Signal Strength as only POI ---    // currently HiggsDecayWidth
        POIs="HiggsDecayWidth"
	self.modelBuilder.doVar("r[1,0,10]")
	if   self.rMode == "poi": 
    		if self.floatMass:
        	    if self.modelBuilder.out.var("MH"):
        	        self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
        	        self.modelBuilder.out.var("MH").setConstant(False)
        	    else:
        	        self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
        	    POIs = "r,MH,HiggsDecayWidth"
        	else:
        	    if self.modelBuilder.out.var("MH"):
        	        self.modelBuilder.out.var("MH").setVal(self.options.mass)
        	        self.modelBuilder.out.var("MH").setConstant(True)
        	    else:
        	        self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
        	    POIs = "r,HiggsDecayWidth"
	elif self.rMode == "nuisance":  
		self.modelBuilder.out.var("r").setAttribute("flatParam")
    		if self.floatMass:
        	    if self.modelBuilder.out.var("MH"):
        	        self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
        	        self.modelBuilder.out.var("MH").setConstant(False)
        	    else:
        	        self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1])) 
        	    POIs = "MH,HiggsDecayWidth"
        	else:
        	    if self.modelBuilder.out.var("MH"):
        	        self.modelBuilder.out.var("MH").setVal(self.options.mass)
        	        self.modelBuilder.out.var("MH").setConstant(True)
        	    else:
        	        self.modelBuilder.doVar("MH[%g]" % self.options.mass) 
	else: raise RuntimeError, "FloatingHiggsWidth: the signal strength must be set to 'poi'(default), 'nuisance'"

        if self.modelBuilder.out.var("HiggsDecayWidth"):
            self.modelBuilder.out.var("HiggsDecayWidth").setRange(float(self.widthRange[0]),float(self.widthRange[1]))
            self.modelBuilder.out.var("HiggsDecayWidth").setConstant(False)
        else:
            self.modelBuilder.doVar("HiggsDecayWidth[%s,%s]" % (self.widthRange[0],self.widthRange[1]))
        self.modelBuilder.doSet("POI",POIs)

    def getHiggsSignalYieldScale(self,production,decay, energy):
            return "r"

floatingHiggsWidth = FloatingHiggsWidth()
