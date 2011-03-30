import ROOT
from sys import stderr, stdout

ROOFIT_EXPR = "expr"
N_OBS_MAX   = 10000

class ModelBuilderBase():
    """This class defines the basic stuff for a model builder, and it's an interface on top of RooWorkspace::factory or HLF files"""
    def __init__(self,options):
        self.options = options
        self.out = stdout
        if options.bin:
            if options.out != None:
                    ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
                    self.out = ROOT.RooWorkspace("w","w");
                    self.out._import = getattr(self.out,"import") # workaround: import is a python keyword
                    self.out.dont_delete = []
            else:
                raise RuntimeError, "You need to specify an output file when using binary mode";
        elif options.out != None:
            #stderr.write("Will save workspace to HLF file %s" % options.out)
            self.out = open(options.out, "w");
        if options.cexpr:
            global ROOFIT_EXPR;
            ROOFIT_EXPR = "cexpr"            
    def factory_(self,X):
        ret = self.out.factory(X);
        if ret: 
            self.out.dont_delete.append(ret)
            return ret
        else:
            print "ERROR parsing '%s'" % X
            self.out.Print("V");
            raise RuntimeError, "Error in factory statement" 
    def doComment(self,X):
        if not self.options.bin: self.out.write("// "+X+"\n");
    def doVar(self,vardef):
        if self.options.bin: self.factory_(vardef);
        else: self.out.write(vardef+";\n");
    def doSet(self,name,vars):
        if self.options.bin: self.out.defineSet(name,vars)
        else: self.out.write("%s = set(%s);\n" % (name,vars));
    def doObj(self,name,type,X):
        if self.options.bin: return self.factory_("%s::%s(%s)" % (type, name, X));
        else: self.out.write("%s = %s(%s);\n" % (name, type, X))

class ModelBuilder(ModelBuilderBase):
    """This class defines the actual methods to build a model"""
    def __init__(self,datacard,options):
        ModelBuilderBase.__init__(self,options) 
        self.DC = datacard
    def doModel(self):
        self.doObservables()
        self.doParametersOfInterest()
        self.doNuisances()
        self.doExpectedEvents()
        self.doIndividualModels()
        self.doCombination()
        if self.options.bin:
            self.doModelConfigs()
    def doObservables(self):
        """create pdf_bin<X> and pdf_bin<X>_bonly for each bin"""
        raise RuntimeError, "Not implemented in ModelBuilder"
    def doParametersOfInterest(self):
        self.doComment(" ----- parameters of interest -----")
        self.doComment(" --- Signal Strength --- ")
        self.doVar("r[0,20]");
        self.doComment(" --- set of all parameters of interest --- ")
        self.doSet("POI","r")
    def doNuisances(self):
        if len(self.DC.systs) == 0: return
        self.doComment(" ----- nuisances -----")
        globalobs = []
        for (n,pdf,args,errline) in self.DC.systs: 
            if pdf == "lnN":
                #print "thetaPdf_%s = Gaussian(theta_%s[-5,5], 0, 1);" % (n,n)
                self.doObj("thetaPdf_%s" % n, "Gaussian", "theta_%s[-5,5], thetaIn_%s[0], 1" % (n,n));
                globalobs.append("thetaIn_%s" % n)
            elif pdf == "gmM":
                val = 0;
                for c in errline.values(): #list channels
                  for v in c.values():     # list effects in each channel 
                    if v != 0:
                        if val != 0 and v != val: 
                            raise RuntimeError, "Error: line %s contains two different uncertainties %g, %g, which is not supported for gmM" % (n,v,val)
                        val = v;
                if val == 0: raise RuntimeError, "Error: line %s contains all zeroes"
                theta = val*val; kappa = 1/theta
                self.doObj("thetaPdf_%s" % n, "Gamma", "theta_%s[1,%f,%f], %g, %g, 0" % (n, max(0.01,1-5*val), 1+5*val, kappa, theta))
            elif pdf == "gmN":
                self.doObj("thetaPdf_%s" % n, "Poisson", "thetaIn_%s[%d], theta_%s[0,%d]" % (n,args[0],n,2*args[0]+5))
                globalobs.append("thetaIn_%s" % n)
        self.doSet("nuisances", ",".join(["theta_%s"    % n for (n,p,a,e) in self.DC.systs]))
        self.doObj("nuisancePdf", "PROD", ",".join(["thetaPdf_%s" % n for (n,p,a,e) in self.DC.systs]))
        if globalobs:
            self.doSet("globalObservables", ",".join(globalobs))
    def doExpectedEvents(self):
        self.doComment(" --- Expected events in each bin, for each process ----")
        for b in self.DC.bins:
            for p in self.DC.exp[b].keys(): # so that we get only self.DC.processes contributing to this bin
                # if it's a zero background, write a zero and move on
                if self.DC.exp[b][p] == 0:
                    self.doVar("n_exp_bin%s_proc_%s[%g]" % (b, p, self.DC.exp[b][p]))
                    continue
                # collect multiplicative corrections
                strexpr = ""; strargs = ""
                gammaNorm = None; iSyst=-1
                if self.DC.isSignal[p]:
                    strexpr += " * @0";
                    strargs += ", r";
                    iSyst += 1
                for (n,pdf,args,errline) in self.DC.systs:
                    if not errline[b].has_key(p): continue
                    if errline[b][p] == 0.0: continue
                    if pdf == "lnN" and errline[b][p] == 1.0: continue
                    iSyst += 1
                    if pdf == "lnN":
                        if type(errline[b][p]) == list:
                            elow, ehigh = errline[b][p];
                            strexpr += " * @%d" % iSyst
                            strargs += ", AsymPow(%f,%f,theta_%s)" % (elow, ehigh, n)
                        else:
                            strexpr += " * pow(%f,@%d)" % (errline[b][p], iSyst)
                            strargs += ", theta_%s" % n
                    elif pdf == "gmM":
                        strexpr += " * @%d " % iSyst
                        strargs += ", theta_%s" % n
                    elif pdf == "gmN":
                        strexpr += " * @%d " % iSyst
                        strargs += ", theta_%s" % n
                        if abs(errline[b][p] * args[0] - self.DC.exp[b][p]) > max(0.05 * max(self.DC.exp[b][p],1), errline[b][p]):
                            raise RuntimeError, "Values of N = %d, alpha = %g don't match with expected rate %g for systematics %s " % (
                                                    args[0], errline[b][p], self.DC.exp[b][p], n)
                        if gammaNorm != None:
                            raise RuntimeError, "More than one gmN uncertainty for the same bin and process (second one is %s)" % n
                        gammaNorm = "%g" % errline[b][p]
                # set base term (fixed or gamma)
                if gammaNorm != None:
                    strexpr = gammaNorm + strexpr
                else:
                    strexpr = str(self.DC.exp[b][p]) + strexpr
                # optimize constants
                if strargs != "":
                    self.doObj("n_exp_bin%s_proc_%s" % (b,p), ROOFIT_EXPR, "'%s'%s" % (strexpr, strargs));
                else:
                    self.doVar("n_exp_bin%s_proc_%s[%g]" % (b, p, self.DC.exp[b][p]))
    def doIndividualModels(self):
        """create pdf_bin<X> and pdf_bin<X>_bonly for each bin"""
        raise RuntimeError, "Not implemented in ModelBuilder"
    def doCombination(self):
        """create model_s and model_b pdfs"""
        raise RuntimeError, "Not implemented in ModelBuilder"
    def doModelConfigs(self):
        if not self.options.bin: raise RuntimeException
        if self.options.out == None: raise RuntimeException
        mc_s = ROOT.RooStats.ModelConfig("ModelConfig",   self.out)
        mc_b = ROOT.RooStats.ModelConfig("ModelConfig_b", self.out)
        for (l,mc) in [ ('s',mc_s), ('b',mc_b) ]:
            mc.SetPdf(self.out.pdf("model_"+l))
            mc.SetParametersOfInterest(self.out.set("POI"))
            mc.SetObservables(self.out.set("observables"))
            if len(self.DC.systs):  mc.SetNuisanceParameters(self.out.set("nuisances"))
            if self.out.set("globalObservables"): mc.SetGlobalObservables(self.out.set("globalObservables"))
            self.out._import(mc, mc.GetName())
        self.out.writeToFile(self.options.out)

class CountingModelBuilder(ModelBuilder):
    """ModelBuilder to make a counting experiment"""
    def __init__(self,datacard,options):
        ModelBuilder.__init__(self,datacard,options)
        if datacard.hasShapes: 
            raise RuntimeError, "You're using a CountingModelBuilder for a model that has shapes"
    def doObservables(self):
        if len(self.DC.obs):
            self.doComment(" ----- observables (already set to asimov values) -----")
            for b in self.DC.bins: self.doVar("n_obs_bin%s[%f,0,%d]" % (b,self.DC.obs[b],N_OBS_MAX))
        else:
            self.doComment(" ----- observables -----")
            for b in self.DC.bins: self.doVar("n_obs_bin%s[0,%d]" % (b,N_OBS_MAX))
        self.doSet("observables", ",".join(["n_obs_bin%s" % b for b in self.DC.bins]))
    def doIndividualModels(self):
        self.doComment(" --- Expected events in each bin, total (S+B and B) ----")
        for b in self.DC.bins:
            self.doObj("n_exp_bin%s_bonly" % b, "sum", ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in self.DC.exp[b].keys() if self.DC.isSignal[p] == False]) )
            self.doObj("n_exp_bin%s"       % b, "sum", ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in self.DC.exp[b].keys()                        ]) )
            self.doObj("pdf_bin%s"       % b, "Poisson", "n_obs_bin%s, n_exp_bin%s"       % (b,b))
            self.doObj("pdf_bin%s_bonly" % b, "Poisson", "n_obs_bin%s, n_exp_bin%s_bonly" % (b,b))
    def doCombination(self):
        prefix = "modelObs" if len(self.DC.systs) else "model" # if no systematics, we build directly the model
        nbins = len(self.DC.bins)
        if nbins > 50:
            from math import ceil
            nblocks = int(ceil(nbins/10.))
            for i in range(nblocks):
                self.doObj("%s_s_%d" % (prefix,i), "PROD", ",".join(["pdf_bin%s"       % self.DC.bins[j] for j in range(10*i,min(nbins,10*i+10))]))
                self.doObj("%s_b_%d" % (prefix,i), "PROD", ",".join(["pdf_bin%s_bonly" % self.DC.bins[j] for j in range(10*i,min(nbins,10*i+10))]))
            self.doObj("%s_s" % prefix, "PROD", ",".join([prefix+"_s_%d" % i for i in range(nblocks)]))
            self.doObj("%s_b" % prefix, "PROD", ",".join([prefix+"_b_%d" % i for i in range(nblocks)]))
        else: 
            self.doObj("%s_s" % prefix, "PROD", ",".join(["pdf_bin%s"       % b for b in self.DC.bins]))
            self.doObj("%s_b" % prefix, "PROD", ",".join(["pdf_bin%s_bonly" % b for b in self.DC.bins]))
        if len(self.DC.systs): # multiply by nuisances if needed
            self.doObj("model_s", "PROD", "modelObs_s, nuisancePdf")
            self.doObj("model_b", "PROD", "modelObs_b, nuisancePdf")

