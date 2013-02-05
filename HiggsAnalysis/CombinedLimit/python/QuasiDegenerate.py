from HiggsAnalysis.CombinedLimit.PhysicsModel import *

#class for testing two near mass-degenerate Higgs model

class QuasiDegenerate(PhysicsModel):
    def __init__(self): 
        self.secondSignal = "_2" #this goes into workspace naming convention
        self.mHRange = ['124','128']
        self.mH = '125';
        self.DMHRange = ['0.01','5']
        self.DMH = '1';
        self.frac = '0.8';
        self.mu = '1';
        self.muAsPOI = False
        self.mHAsPOI = False
        self.DMHAsPOI = False
        self.fracAsPOI = False
        self.floatFrac = True
        self.floatMH = True
        self.floatDMH = True
        self.floatMu = True
    def setModelBuilder(self, modelBuilder):
        PhysicsModel.setModelBuilder(self, modelBuilder)
        self.modelBuilder.doModelBOnly = False
    
    def setPhysicsOptions(self,physOptions):
        for po in physOptions:
            if po.startswith("higgsMassRange="):
                self.mHRange = po.replace("higgsMassRange=","").split(",")
                print 'The Higgs mass range:', self.mHRange
                if len(self.mHRange) != 2:
                    raise RuntimeError, "Higgs mass range definition requires two extrema"
                elif float(self.mHRange[0]) >= float(self.mHRange[1]):
                    raise RuntimeError, "Extrema for Higgs mass range defined with inverterd order. Second must be larger the first"
            if po.startswith("DMHRange="):
                self.DMHRange = po.replace("DMHRange=","").split(",")
                print 'The Higgs mass range:', self.DMHRange
                if len(self.DMHRange) != 2:
                    raise RuntimeError, "DMH range definition requires two extrema"
                elif float(self.DMHRange[0]) >= float(self.DMHRange[1]):
                    raise RuntimeError, "Extrema for DMH range defined with inverterd order. Second must be larger the first"
            
            if po == "muAsPOI": 
                print "Will consider the signal strength as a parameter of interest"
                self.muAsPOI = True
                self.floatMu = True
            if po.startswith("fixMu="):
                self.floatMu = False
                self.muAsPOI = False
                self.mu = po.replace("fixMu=","")
                print "will set mu to be %s" % self.mu
            if po == "mHAsPOI": 
                print "Will consider the mass 1 as a parameter of interest"
                self.mHAsPOI = True
                self.floatMH = True
            if po.startswith("fixMH="): 
                self.floatMH = False
                self.mHAsPOI = False
                self.mH = po.replace("fixMH=","")
                print "will set mH to be %s" % self.mH


            if po == "DMHAsPOI": 
                print "Will consider the DMH  as a parameter of interest"
                self.DMHAsPOI = True
                self.floatDMH = True
            if po.startswith("fixDMH="): 
                self.floatDMH = False
                self.DMHAsPOI = False
                self.DMH = po.replace("fixDMH=","")
                print "will set DMH to be %s" % self.DMH

  
            if po == "fracAsPOI": 
                print "Will consider the frac  as a parameter of interest"
                self.fracAsPOI = True
                self.floatFrac = True
            if po.startswith("fixFrac="): 
                self.floatFrac = False
                self.fracAsPOI = False
                self.frac = po.replace("fixFrac=","")
                print "will set frac to be %s" % self.frac


    def doParametersOfInterest(self):
        poi=""
###############################Frac	
        self.modelBuilder.doVar("x[0.8,0.0,1]"); #so first Higgs always has the bigger fraction 
        if self.floatFrac: 
          if self.fracAsPOI: poi += "x"  
        else:
          self.modelBuilder.out.var("x").setVal(float(self.frac))   #Test for one Higgs
          self.modelBuilder.out.var("x").setConstant(True)

        self.modelBuilder.out.var("x").Print("")

################################DMH
        if self.modelBuilder.out.var("DMH"):
            if self.floatDMH:
              self.modelBuilder.out.var("DMH").setRange(float(self.DMHRange[0]),float(self.DMHRange[1]))
              self.modelBuilder.out.var("DMH").setConstant(False)
              if self.DMHAsPOI: poi += ",DMH"
            else:
              self.modelBuilder.out.var("DMH").setVal(float(self.DMH))
              self.modelBuilder.out.var("DMH").setConstant(True)
        else:
            self.modelBuilder.doVar("DMH[%s,%s,%s]" % (self.DMH, self.DMHRange[0],self.DMHRange[1]));
            if self.floatDMH:
              if self.DMHAsPOI: poi += ",DMH"
            else:
              self.modelBuilder.out.var("DMH").setVal(float(self.DMH))
              self.modelBuilder.out.var("DMH").setConstant(True)



################################Mu
        if self.floatMu:
           self.modelBuilder.doVar("r[1,0,10]");
           if self.muAsPOI: poi += ",r"
        else:  
           self.modelBuilder.doVar("r[%g]" % self.mu);
           self.modelBuilder.out.var("r").setConstant(True)
#           self.modelBuilder.factory_("expr::not_x(\"(1-@1)\", x)")
#           self.sigNorms = { True:'x', False:'not_x' }
        
        self.modelBuilder.factory_("expr::r_times_not_x(\"@0*(1-@1)\", r, x)")
        self.modelBuilder.factory_("expr::r_times_x(\"@0*@1\", r, x)")
        self.sigNorms = { True:'r_times_not_x', False:'r_times_x' } #MH: r*x  MH_2: r*(1-x)


        if self.floatMH:
            if self.modelBuilder.out.var("MH"):
                self.modelBuilder.out.var("MH").setRange(float(self.mHRange[0]),float(self.mHRange[1]))
                self.modelBuilder.out.var("MH").setConstant(False)
            else:
                self.modelBuilder.doVar("MH[%s,%s]" % (self.mHRange[0],self.mHRange[1]))
            if self.mHAsPOI: poi += ",MH"
        else:
          if self.modelBuilder.out.var("MH"):
            self.modelBuilder.out.var("MH").setVal(float(self.mH))
          else:
            self.modelBuilder.doVar("MH[%g]" % self.mH)
          self.modelBuilder.out.var("MH").setConstant(True)

        if poi.startswith(","): print poi; poi = poi.replace(",","",1)
        self.modelBuilder.doSet("POI",poi)
        print "pois %s" %poi		



    def getYieldScale(self,bin,process):
         if not self.DC.isSignal[process]: return 1
         return self.sigNorms[self.secondSignal in process]

quasiDegenerate = QuasiDegenerate()

