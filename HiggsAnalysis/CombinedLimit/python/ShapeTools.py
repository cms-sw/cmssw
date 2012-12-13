from sys import stdout, stderr
import os.path
import ROOT

from HiggsAnalysis.CombinedLimit.ModelTools import ModelBuilder

class ShapeBuilder(ModelBuilder):
    def __init__(self,datacard,options):
        ModelBuilder.__init__(self,datacard,options) 
        if not datacard.hasShapes: 
            raise RuntimeError, "You're using a ShapeBuilder for a model that has no shapes"
        if options.libs:
            for lib in options.libs:
                ROOT.gSystem.Load(lib)
    	self.wspnames = {}
    	self.wsp = None
    ## ------------------------------------------
    ## -------- ModelBuilder interface ----------
    ## ------------------------------------------
    def doObservables(self):
        if (self.options.verbose > 1): stderr.write("Using shapes: qui si parra' la tua nobilitate\n")
        self.prepareAllShapes();
        if len(self.DC.bins) > 1 or not self.options.forceNonSimPdf:
            ## start with just a few channels
            strexpr="CMS_channel[" + ",".join(["%s=%d" % (l,i) for i,l in enumerate(self.DC.bins[:5])]) + "]";
            self.doVar(strexpr);
            self.out.binCat = self.out.cat("CMS_channel");
            ## then add all the others, to avoid a too long factory string
            for i,l in enumerate(self.DC.bins[5:]): self.out.binCat.defineType(l,i+5)   
            if self.options.verbose: stderr.write("Will use category 'CMS_channel' to identify the %d channels\n" % self.out.binCat.numTypes())
            self.out.obs = ROOT.RooArgSet()
            self.out.obs.add(self.out.binVars)
            self.out.obs.add(self.out.binCat)
        else:
            self.out.obs = self.out.binVars
        self.doSet("observables",self.out.obs)
        if len(self.DC.obs) != 0: 
            self.doCombinedDataset()
    def doIndividualModels(self):
        for b in self.DC.bins:
            pdfs   = ROOT.RooArgList(); bgpdfs   = ROOT.RooArgList()
            coeffs = ROOT.RooArgList(); bgcoeffs = ROOT.RooArgList()
            for p in self.DC.exp[b].keys(): # so that we get only self.DC.processes contributing to this bin
                if self.DC.exp[b][p] == 0: continue
                (pdf,coeff) = (self.getPdf(b,p), self.out.function("n_exp_bin%s_proc_%s" % (b,p)))
                extranorm = self.getExtraNorm(b,p)
                if extranorm:
                    prodset = ROOT.RooArgList(self.out.function("n_exp_bin%s_proc_%s" % (b,p)))
                    for X in extranorm: prodset.add(self.out.function(X))
                    prodfunc = ROOT.RooProduct("n_exp_final_bin%s_proc_%s" % (b,p), "", prodset)
                    self.out._import(prodfunc)
                    coeff = self.out.function("n_exp_final_bin%s_proc_%s" % (b,p))                    
                pdfs.add(pdf); coeffs.add(coeff)
                if not self.DC.isSignal[p]:
                    bgpdfs.add(pdf); bgcoeffs.add(coeff)
            if self.options.verbose: print "Creating RooAddPdf %s with %s elements" % ("pdf_bin"+b, coeffs.getSize())
            sum_s = ROOT.RooAddPdf("pdf_bin%s"       % b, "",   pdfs,   coeffs)
            sum_b = ROOT.RooAddPdf("pdf_bin%s_bonly" % b, "", bgpdfs, bgcoeffs)
            if b in self.pdfModes: 
                sum_s.setAttribute('forceGen'+self.pdfModes[b].title())
                sum_b.setAttribute('forceGen'+self.pdfModes[b].title())
            if len(self.DC.systs):
                ## rename the pdfs
                sum_s.SetName("pdf_bin%s_nuis" % b); sum_b.SetName("pdf_bin%s_bonly_nuis" % b)
                # now we multiply by all the nuisances, but avoiding nested products
                # so we first make a list of all nuisances plus the RooAddPdf
                sumPlusNuis_s = ROOT.RooArgList(self.out.nuisPdfs); sumPlusNuis_s.add(sum_s)
                sumPlusNuis_b = ROOT.RooArgList(self.out.nuisPdfs); sumPlusNuis_b.add(sum_b)
                # then make RooProdPdf and import it
                pdf_s = ROOT.RooProdPdf("pdf_bin%s"       % b, "", sumPlusNuis_s) 
                pdf_b = ROOT.RooProdPdf("pdf_bin%s_bonly" % b, "", sumPlusNuis_b) 
                if b in self.pdfModes: 
                    pdf_s.setAttribute('forceGen'+self.pdfModes[b].title())
                    pdf_b.setAttribute('forceGen'+self.pdfModes[b].title())
                self.out._import(pdf_s, ROOT.RooFit.RenameConflictNodes(b))
                self.out._import(pdf_b, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
            else:
                self.out._import(sum_s, ROOT.RooFit.RenameConflictNodes(b))
                self.out._import(sum_b, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    def doCombination(self):
        ## Contrary to Number-counting models, here each channel PDF already contains the nuisances
        ## So we just have to build the combined pdf
        if len(self.DC.bins) > 1 or not self.options.forceNonSimPdf:
            for (postfixIn,postfixOut) in [ ("","_s"), ("_bonly","_b") ]:
                simPdf = ROOT.RooSimultaneous("model"+postfixOut, "model"+postfixOut, self.out.binCat)
                for b in self.DC.bins:
                    pdfi = self.out.pdf("pdf_bin%s%s" % (b,postfixIn))
                    simPdf.addPdf(pdfi, b)
                self.out._import(simPdf)
        else:
            self.out._import(self.out.pdf("pdf_bin%s"       % self.DC.bins[0]).clone("model_s"), ROOT.RooFit.Silence())
            self.out._import(self.out.pdf("pdf_bin%s_bonly" % self.DC.bins[0]).clone("model_b"), ROOT.RooFit.Silence())
        if self.options.fixpars:
            pars = self.out.pdf("model_s").getParameters(self.out.obs)
            iter = pars.createIterator()
            while True:
                arg = iter.Next()
                if arg == None: break;
                if arg.InheritsFrom("RooRealVar") and arg.GetName() != "r": 
                    arg.setConstant(True);
    ## --------------------------------------
    ## -------- High level helpers ----------
    ## --------------------------------------
    def prepareAllShapes(self):
        shapeTypes = []; shapeBins = []; shapeObs = {}
        self.pdfModes = {}
        for ib,b in enumerate(self.DC.bins):
            databins = {}; bgbins = {}
            for p in [self.options.dataname]+self.DC.exp[b].keys():
                if len(self.DC.obs) == 0 and p == self.options.dataname: continue
                if p != self.options.dataname and self.DC.exp[b][p] == 0: continue
                shape = self.getShape(b,p); norm = 0;
                if shape == None: # counting experiment
                    if not self.out.var("CMS_fakeObs"): 
                        self.doVar("CMS_fakeObs[0,1]");
                        self.out.var("CMS_fakeObs").setBins(1);
                        self.doSet("CMS_fakeObsSet","CMS_fakeObs");
                        shapeObs["CMS_fakeObsSet"] = self.out.set("CMS_fakeObsSet")
                    if p == self.options.dataname:
                        self.pdfModes[b] = 'binned'
                        shapeTypes.append("RooDataHist")
                    else:
                        shapeTypes.append("RooAbsPdf");
                elif shape.ClassName().startswith("TH1"):
                    shapeTypes.append("TH1"); shapeBins.append(shape.GetNbinsX())
                    norm = shape.Integral()
                    if p == self.options.dataname: 
                        if self.options.poisson > 0 and norm > self.options.poisson:
                            self.pdfModes[b] = 'poisson'
                        else:
                            self.pdfModes[b] = 'binned'
                        for i in xrange(1, shape.GetNbinsX()+1):
                            if shape.GetBinContent(i) > 0: databins[i] = True
                    elif not self.DC.isSignal[p]:
                        for i in xrange(1, shape.GetNbinsX()+1):
                            if shape.GetBinContent(i) > 0: bgbins[i] = True
                elif shape.InheritsFrom("RooDataHist"):
                    shapeTypes.append("RooDataHist"); 
                    shapeBins.append(shape.numEntries())
                    shapeObs[self.argSetToString(shape.get())] = shape.get()
                    norm = shape.sumEntries()
                    if p == self.options.dataname: 
                        if self.options.poisson > 0 and norm > self.options.poisson:
                            self.pdfModes[b] = 'poisson'
                        else:
                            self.pdfModes[b] = 'binned'
                elif shape.InheritsFrom("RooDataSet"):
                    shapeTypes.append("RooDataSet"); 
                    shapeObs[self.argSetToString(shape.get())] = shape.get()
                    norm = shape.sumEntries()
                    if p == self.options.dataname: self.pdfModes[b] = 'unbinned'
                elif shape.InheritsFrom("TTree"):
                    shapeTypes.append("TTree"); 
                    if p == self.options.dataname: self.pdfModes[b] = 'unbinned'
                elif shape.InheritsFrom("RooAbsPdf"):
                    shapeTypes.append("RooAbsPdf");
                else: raise RuntimeError, "Currently supporting only TH1s, RooDataHist and RooAbsPdfs"
                if norm != 0:
                    if p == self.options.dataname:
                        if len(self.DC.obs):
                            if self.DC.obs[b] == -1: self.DC.obs[b] = norm
                            elif self.DC.obs[b] == 0 and norm > 0.01:
                                if not self.options.noCheckNorm: raise RuntimeError, "Mismatch in normalizations for observed data in bin %s: text %f, shape %f" % (b,self.DC.obs[b],norm)
                            elif self.DC.obs[b] >0 and abs(norm/self.DC.obs[b]-1) > 0.005:
                                if not self.options.noCheckNorm: raise RuntimeError, "Mismatch in normalizations for observed data in bin %s: text %f, shape %f" % (b,self.DC.obs[b],norm)
                    else:
                        if self.DC.exp[b][p] == -1: self.DC.exp[b][p] = norm
                        elif self.DC.exp[b][p] > 0 and abs(norm-self.DC.exp[b][p]) > 0.01*max(1,self.DC.exp[b][p]): 
                            if not self.options.noCheckNorm: raise RuntimeError, "Mismatch in normalizations for bin %s, process %s: rate %f, shape %f" % (b,p,self.DC.exp[b][p],norm)
            if len(databins) > 0:
                for i in databins.iterkeys():
                    if i not in bgbins: stderr.write("Channel %s has bin %d fill in data but empty in all backgrounds\n" % (b,i))
        if shapeTypes.count("TH1"):
            self.out.maxbins = max(shapeBins)
            if self.options.verbose: stderr.write("Will use binning variable CMS_th1x with %d bins\n" % self.out.maxbins)
            self.doVar("CMS_th1x[0,%d]" % self.out.maxbins); self.out.var("CMS_th1x").setBins(self.out.maxbins)
            self.out.binVar = self.out.var("CMS_th1x")
            shapeObs['CMS_th1x'] = self.out.binVar
        if shapeTypes.count("TH1") == len(shapeTypes):
            self.out.mode    = "binned"
            self.out.binVars = ROOT.RooArgSet(self.out.binVar)
        elif shapeTypes.count("RooDataSet") > 0 or shapeTypes.count("TTree") > 0 or len(shapeObs.keys()) > 1:
            self.out.mode = "unbinned"
            if self.options.verbose: stderr.write("Will try to work with unbinned datasets\n")
            if self.options.verbose: stderr.write("Observables: %s\n" % str(shapeObs.keys()))
            if len(shapeObs.keys()) != 1:
                self.out.binVars = ROOT.RooArgSet()
                for obs in shapeObs.values():
                     self.out.binVars.add(obs, False)
            else:
                self.out.binVars = shapeObs.values()[0]
            self.out._import(self.out.binVars)
        else:
            self.out.mode = "binned"
            if self.options.verbose: stderr.write("Will try to make a binned dataset\n")
            if self.options.verbose: stderr.write("Observables: %s\n" % str(shapeObs.keys()))
            if len(shapeObs.keys()) != 1:
                raise RuntimeError, "There's more than once choice of observables: %s\n" % str(shapeObs.keys())
            self.out.binVars = shapeObs.values()[0]
            self.out._import(self.out.binVars)
    def doCombinedDataset(self):
        if len(self.DC.bins) == 1 and self.options.forceNonSimPdf:
            data = self.getData(self.DC.bins[0],self.options.dataname).Clone(self.options.dataname)
            self.out._import(data)
            return
        if self.out.mode == "binned":
            combiner = ROOT.CombDataSetFactory(self.out.obs, self.out.binCat)
            for b in self.DC.bins: combiner.addSetBin(b, self.getData(b,self.options.dataname))
            self.out.data_obs = combiner.done(self.options.dataname,self.options.dataname)
            self.out._import(self.out.data_obs)
        elif self.out.mode == "unbinned":
            combiner = ROOT.CombDataSetFactory(self.out.obs, self.out.binCat)
            for b in self.DC.bins: combiner.addSetAny(b, self.getData(b,self.options.dataname))
            self.out.data_obs = combiner.doneUnbinned(self.options.dataname,self.options.dataname)
            self.out._import(self.out.data_obs)
        else: raise RuntimeException, "Only combined datasets are supported"
        #print "Created combined dataset with ",self.out.data_obs.numEntries()," entries, out of:"
        #for b in self.DC.bins: print "  bin", b, ": entries = ", self.getData(b,self.options.dataname).numEntries()
    ## -------------------------------------
    ## -------- Low level helpers ----------
    ## -------------------------------------
    def getShape(self,channel,process,syst="",_fileCache={},_cache={},allowNoSyst=False):
        if _cache.has_key((channel,process,syst)): 
            if self.options.verbose > 1: print "recyling (%s,%s,%s) -> %s\n" % (channel,process,syst,_cache[(channel,process,syst)].GetName())
            return _cache[(channel,process,syst)];
        postFix="Sig" if (process in self.DC.isSignal and self.DC.isSignal[process]) else "Bkg"
        bentry = None
        if self.DC.shapeMap.has_key(channel): bentry = self.DC.shapeMap[channel]
        elif self.DC.shapeMap.has_key("*"):   bentry = self.DC.shapeMap["*"]
        else: raise KeyError, "Shape map has no entry for channel '%s'" % (channel)
        names = []
        if bentry.has_key(process): names = bentry[process]
        elif bentry.has_key("*"):   names = bentry["*"]
        elif self.DC.shapeMap["*"].has_key(process): names = self.DC.shapeMap["*"][process]
        elif self.DC.shapeMap["*"].has_key("*"):     names = self.DC.shapeMap["*"]["*"]
        else: raise KeyError, "Shape map has no entry for process '%s', channel '%s'" % (process,channel)
        if len(names) == 1 and names[0] == "FAKE": return None
        if syst != "": 
            if len(names) == 2:
                if allowNoSyst: return None
                raise RuntimeError, "Can't find systematic "+syst+" for process '%s', channel '%s'" % (process,channel)
            names = [names[0], names[2]]
        else:   
            names = [names[0], names[1]]
        strmass = "%d" % self.options.mass if self.options.mass % 1 == 0 else str(self.options.mass)
        finalNames = [ x.replace("$PROCESS",process).replace("$CHANNEL",channel).replace("$SYSTEMATIC",syst).replace("$MASS",strmass) for x in names ]
        if not _fileCache.has_key(finalNames[0]): 
            trueFName = finalNames[0]
            if not os.path.exists(trueFName) and not os.path.isabs(trueFName) and os.path.exists(self.options.baseDir+"/"+trueFName):
                trueFName = self.options.baseDir+"/"+trueFName;
            _fileCache[finalNames[0]] = ROOT.TFile.Open(trueFName)
        file = _fileCache[finalNames[0]]; objname = finalNames[1]
        if not file: raise RuntimeError, "Cannot open file %s (from pattern %s)" % (finalNames[0],names[0])
        if ":" in objname: # workspace:obj or ttree:xvar or th1::xvar
            (wname, oname) = objname.split(":")
            if (file,wname) not in self.wspnames : 
		self.wspnames[(file,wname)] = file.Get(wname)
	    self.wsp = self.wspnames[(file,wname)]
            if not self.wsp: raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (wname,finalNames[0],names[1],names[0])
            if self.wsp.ClassName() == "RooWorkspace":
                ret = self.wsp.data(oname)
                if not ret: ret = self.wsp.pdf(oname)
                if not ret: raise RuntimeError, "Object %s in workspace %s in file %s does not exist or it's neither a data nor a pdf" % (oname, wname, finalNames[0])
                # Fix the fact that more than one entry can refer to the same object
                ret = ret.Clone()
                ret.SetName("shape%s_%s_%s%s" % (postFix,process,channel, "_"+syst if syst else ""))
                _cache[(channel,process,syst)] = ret
                if not syst:
                  normname = "%s_norm" % (oname)
                  norm = self.wsp.arg(normname)
                  if norm: 
                    if normname in self.DC.flatParamNuisances: 
                        self.DC.flatParamNuisances[normname] = False # don't warn if not found
                        norm.setAttribute("flatParam")
                    norm.SetName("shape%s_%s_%s%s_norm" % (postFix,process,channel, "_"))
                    self.out._import(norm, ROOT.RooFit.RecycleConflictNodes()) 
                if self.options.verbose > 1: print "import (%s,%s) -> %s\n" % (finalNames[0],objname,ret.GetName())
                return ret;
            elif self.wsp.ClassName() == "TTree":
                ##If it is a tree we will convert it in RooDataSet . Then we can decide if we want to build a
                ##RooKeysPdf or if we want to use it as an unbinned dataset 
                if not self.wsp: raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (wname,finalNames[0],names[1],names[0])
                self.doVar("%s[%f,%f]" % (oname,self.wsp.GetMinimum(oname),self.wsp.GetMaximum(oname)))
                #Check if it is weighted
                self.doVar("__WEIGHT__[0.,1000.]")
                rds = ROOT.RooDataSet("shape%s_%s_%s%s" % (postFix,process,channel, "_"+syst if syst else ""), "shape%s_%s_%s%s" % (postFix,process,channel, "_"+syst if syst else ""),self.wsp,ROOT.RooArgSet(self.out.var(oname)),"","__WEIGHT__")
                rds.var = oname
                _cache[(channel,process,syst)] = rds
                if self.options.verbose > 1: print "import (%s,%s) -> %s\n" % (finalNames[0],wname,rds.GetName())
                return rds
            elif self.wsp.InheritsFrom("TH1"):
                ##If it is a Histogram we will convert it in RooDataSet preserving the bins 
                if not self.wsp: raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (wname,finalNames[0],names[1],names[0])
                name = "shape%s_%s_%s%s" % (postFix,process,channel, "_"+syst if syst else "")
                # don't make it twice
                for X in _neverDelete:
                    if X.InheritsFrom("TNamed") and X.GetName() == name: return X
                self.doVar("%s[%f,%f]" % (oname,self.wsp.GetXaxis().GetXmin(),self.wsp.GetXaxis().GetXmax()))
                rds = ROOT.RooDataHist(name, name, ROOT.RooArgList(self.out.var(oname)), self.wsp)
                rds.var = oname
                if self.options.verbose > 1: stderr.write("import (%s,%s) -> %s\n" % (finalNames[0],wname,rds.GetName()))
                _neverDelete.append(rds)
                return rds
            else:
                raise RuntimeError, "Object %s in file %s has unrecognized type %s" (wname, finalNames[0], self.wsp.ClassName())
        else: # histogram
            ret = file.Get(objname);
            if not ret: 
                if allowNoSyst: return None
                raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (objname,finalNames[0],names[1],names[0])
            ret.SetName("shape%s_%s_%s%s" % (postFix,process,channel, "_"+syst if syst else ""))
            if self.options.verbose > 1: print "import (%s,%s) -> %s\n" % (finalNames[0],objname,ret.GetName())
            _cache[(channel,process,syst)] = ret
            return ret
    def getData(self,channel,process,syst="",_cache={}):
        return self.shape2Data(self.getShape(channel,process,syst),channel,process)
    def getPdf(self,channel,process,_cache={}):
        postFix="Sig" if (process in self.DC.isSignal and self.DC.isSignal[process]) else "Bkg"
        if _cache.has_key((channel,process)): return _cache[(channel,process)]
        shapeNominal = self.getShape(channel,process)
        nominalPdf = self.shape2Pdf(shapeNominal,channel,process)
        if shapeNominal == None: return nominalPdf # no point morphing a fake shape
        morphs = []; shapeAlgo = None
        for (syst,nofloat,pdf,args,errline) in self.DC.systs:
            if not "shape" in pdf: continue
            if errline[channel][process] == 0: continue
            allowNoSyst = (pdf[-1] == "?")
            pdf = pdf.replace("?","")
            if shapeAlgo == None:  shapeAlgo = pdf
            elif pdf != shapeAlgo: 
                errmsg =  "ERROR for channel %s, process %s. " % (channel,process)
                errmsg += "Requesting morphing %s  for systematic %s after having requested %s. " % (pdf, syst, shapeAlgo)
                raise RuntimeError, errmsg+" One can use only one morphing algorithm for a given shape";
            if errline[channel][process] != 0:
                if allowNoSyst and not self.isShapeSystematic(channel,process,syst): continue
                shapeUp   = self.getShape(channel,process,syst+"Up")
                shapeDown = self.getShape(channel,process,syst+"Down")
                if shapeUp.ClassName()   != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst %s" % (channel,process,syst)
                if shapeDown.ClassName() != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst %s" % (channel,process,syst)
                morphs.append((syst,errline[channel][process],self.shape2Pdf(shapeUp,channel,process),self.shape2Pdf(shapeDown,channel,process)))
        if len(morphs) == 0: return nominalPdf
        if shapeAlgo == "shapeN": stderr.write("Warning: the shapeN implementation in RooStats and L&S are different\n")
        pdfs = ROOT.RooArgList(nominalPdf)
        coeffs = ROOT.RooArgList()
        minscale = 1
        for (syst,scale,pdfUp,pdfDown) in morphs:
            pdfs.add(pdfUp); pdfs.add(pdfDown);
            if scale == 1:
                coeffs.add(self.out.var(syst))
            else: # must scale it :-/
                coeffs.add(self.doObj("%s_scaled_%s_%s" % (syst,channel,process), "prod","%s, %s" % (scale,syst)))
                if scale < minscale: minscale = scale
        qrange = minscale; qalgo = 0;
        if shapeAlgo[-1] == "*": 
            qalgo = 100
            shapeAlgo = shapeAlgo[:-1]
        if shapeAlgo == "shape": shapeAlgo = self.options.defMorph
        if "shapeL" in shapeAlgo: qrange = 0;
        elif "shapeN" in shapeAlgo: qalgo = -1;
        if "2a" in shapeAlgo: # old shape2
            if not nominalPdf.InheritsFrom("RooHistPdf"):  raise RuntimeError, "Algorithms 'shape2', 'shapeL2', shapeN2' only work with histogram templates"
            if nominalPdf.dataHist().get().getSize() != 1: raise RuntimeError, "Algorithms 'shape2', 'shapeL2', shapeN2' only work in one dimension"
            xvar = nominalPdf.dataHist().get().first()
            _cache[(channel,process)] = ROOT.VerticalInterpHistPdf("shape%s_%s_%s_morph" % (postFix,channel,process), "", xvar, pdfs, coeffs, qrange, qalgo)
        elif "2" in shapeAlgo:  # new faster shape2
            if not nominalPdf.InheritsFrom("RooHistPdf"):  raise RuntimeError, "Algorithms 'shape2', 'shapeL2', shapeN2' only work with histogram templates"
            if nominalPdf.dataHist().get().getSize() != 1: raise RuntimeError, "Algorithms 'shape2', 'shapeL2', shapeN2' only work in one dimension"
            xvar = nominalPdf.dataHist().get().first()
            _cache[(channel,process)] = ROOT.FastVerticalInterpHistPdf("shape%s_%s_%s_morph" % (postFix,channel,process), "", xvar, pdfs, coeffs, qrange, qalgo)
        else:
            _cache[(channel,process)] = ROOT.VerticalInterpPdf("shape%s_%s_%s_morph" % (postFix,channel,process), "", pdfs, coeffs, qrange, qalgo)
        return _cache[(channel,process)]
    def isShapeSystematic(self,channel,process,syst):
        shapeUp = self.getShape(channel,process,syst+"Up",allowNoSyst=True)    
        return shapeUp != None
    def getExtraNorm(self,channel,process):
        postFix="Sig" if (process in self.DC.isSignal and self.DC.isSignal[process]) else "Bkg"
        terms = []
        shapeNominal = self.getShape(channel,process)
        if shapeNominal == None: 
            # FIXME no extra norm for dummy pdfs (could be changed)
            return None
        if shapeNominal.InheritsFrom("RooAbsPdf"): 
            # return nominal multiplicative normalization constant
            normname = "shape%s_%s_%s%s_norm" % (postFix,process,channel, "_")
            if self.out.arg(normname): return [ normname ]
            else: return None
        normNominal = 0
        if shapeNominal.InheritsFrom("TH1"): normNominal = shapeNominal.Integral()
        elif shapeNominal.InheritsFrom("RooDataHist"): normNominal = shapeNominal.sumEntries()
        else: return None    
        if normNominal == 0: raise RuntimeError, "Null norm for channel %s, process %s" % (channel,process)
        for (syst,nofloat,pdf,args,errline) in self.DC.systs:
            if "shape" not in pdf: continue
            if errline[channel][process] != 0:
                if pdf[-1] == "?" and not self.isShapeSystematic(channel,process,syst): continue
                shapeUp   = self.getShape(channel,process,syst+"Up")
                shapeDown = self.getShape(channel,process,syst+"Down")
                if shapeUp.ClassName()   != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                if shapeDown.ClassName() != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                kappaUp,kappaDown = 1,1
                if shapeNominal.InheritsFrom("TH1"):
                    kappaUp,kappaDown = shapeUp.Integral(),shapeDown.Integral()
                elif shapeNominal.InheritsFrom("RooDataHist"):
                    kappaUp,kappaDown = shapeUp.sumEntries(),shapeDown.sumEntries()
                if not kappaUp > 0: raise RuntimeError, "Bogus norm %r for channel %s, process %s, systematic %s Up" % (kappaUp, channel,process,syst)
                if not kappaDown > 0: raise RuntimeError, "Bogus norm %r for channel %s, process %s, systematic %s Down" % (kappaDown, channel,process,syst)
                kappaUp /=normNominal; kappaDown /= normNominal
                if abs(kappaUp-1) < 1e-3 and abs(kappaDown-1) < 1e-3: continue
                # if errline[channel][process] == <x> it means the gaussian should be scaled by <x> before doing pow
                # for convenience, we scale the kappas
                kappasScaled = [ pow(x, errline[channel][process]) for x in kappaDown,kappaUp ]
                self.doObj( "systeff_%s_%s_%s" % (channel,process,syst), "AsymPow", "%f,%f,%s" % (kappasScaled[0], kappasScaled[1], syst) ) 
                terms.append( "systeff_%s_%s_%s" % (channel,process,syst) )
        return terms if terms else None;
    def shape2Data(self,shape,channel,process,_cache={}):
        postFix="Sig" if (process in self.DC.isSignal and self.DC.isSignal[process]) else "Bkg"
        if shape == None:
            name = "shape%s_%s_%s" % (postFix,channel,process)
            if not _cache.has_key(name):
                obs = ROOT.RooArgSet(self.out.var("CMS_fakeObs"))
                obs.setRealValue("CMS_fakeObs",0.5);
                if self.out.mode == "binned":
                    self.out.var("CMS_fakeObs").setBins(1)
                    rdh = ROOT.RooDataHist(name, name, obs)
                    rdh.set(obs, self.DC.obs[channel])
                    _cache[name] = rdh
                else:
                    rds = ROOT.RooDataSet(name, name, obs)
                    if self.DC.obs[channel] == float(int(self.DC.obs[channel])):
                        for i in range(int(self.DC.obs[channel])): rds.add(obs)
                    else:
                        rds.add(obs, self.DC.obs[channel])
                    _cache[name] = rds
            return _cache[name]
        if not _cache.has_key(shape.GetName()):
            if shape.ClassName().startswith("TH1"):
                rebinh1 = ROOT.TH1F(shape.GetName()+"_rebin", "", self.out.maxbins, 0.0, float(self.out.maxbins))
                for i in range(1,min(shape.GetNbinsX(),self.out.maxbins)+1): 
                    rebinh1.SetBinContent(i, shape.GetBinContent(i))
                rdh = ROOT.RooDataHist(shape.GetName(), shape.GetName(), ROOT.RooArgList(self.out.binVar), rebinh1)
                self.out._import(rdh)
                _cache[shape.GetName()] = rdh
            elif shape.ClassName() in ["RooDataHist", "RooDataSet"]:
                return shape
            else: raise RuntimeError, "shape2Data not implemented for %s" % shape.ClassName()
        return _cache[shape.GetName()]
    def shape2Pdf(self,shape,channel,process,_cache={}):
        postFix="Sig" if (process in self.DC.isSignal and self.DC.isSignal[process]) else "Bkg"
        if shape == None:
            name = "shape%s_%s_%s" % (postFix,channel,process)
            if not _cache.has_key(name):
                _cache[name] = ROOT.RooUniform(name, name, ROOT.RooArgSet(self.out.var("CMS_fakeObs")))
            return _cache[name]
        if not _cache.has_key(shape.GetName()+"Pdf"):
            if shape.ClassName().startswith("TH1"):
                rdh = self.shape2Data(shape,channel,process)
                rhp = self.doObj("%sPdf" % shape.GetName(), "HistPdf", "{%s}, %s" % (self.out.binVar.GetName(), shape.GetName()))
                _cache[shape.GetName()+"Pdf"] = rhp
            elif shape.InheritsFrom("RooAbsPdf"):
                _cache[shape.GetName()+"Pdf"] = shape
            elif shape.InheritsFrom("RooDataHist"):
                rhp = ROOT.RooHistPdf("%sPdf" % shape.GetName(), "", shape.get(), shape) 
                self.out._import(rhp)
                _cache[shape.GetName()+"Pdf"] = rhp
            elif shape.InheritsFrom("RooDataSet"):
                rkp = ROOT.RooKeysPdf("%sPdf" % shape.GetName(), "", self.out.var(shape.var), shape,3,1.5); 
                self.out._import(rkp)
                _cache[shape.GetName()+"Pdf"] = rkp
            else: 
                raise RuntimeError, "shape2Pdf not implemented for %s" % shape.ClassName()
        return _cache[shape.GetName()+"Pdf"]
    def argSetToString(self,argset):
        names = []
        it = argset.createIterator()
        while True:
            arg = it.Next()
            if not arg: break
            names.append(arg.GetName())
        return ",".join(names)

