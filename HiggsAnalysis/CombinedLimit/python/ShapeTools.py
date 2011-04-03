from sys import stdout, stderr
import ROOT

from HiggsAnalysis.CombinedLimit.ModelTools import ModelBuilder

class ShapeBuilder(ModelBuilder):
    def __init__(self,datacard,options):
        ModelBuilder.__init__(self,datacard,options) 
        if not datacard.hasShapes: 
            raise RuntimeError, "You're using a ShapeBuilder for a model that has no shapes"
    ## ------------------------------------------
    ## -------- ModelBuilder interface ----------
    ## ------------------------------------------
    def doObservables(self):
        if (self.options.verbose > 1): stderr.write("Using shapes: qui si parra' la tua nobilitate\n")
        self.prepareAllShapes();
        if len(self.DC.bins) > 1:
            strexpr="channel[" + ",".join(["%s=%d" % (l,i) for i,l in enumerate(self.DC.bins)]) + "]";
            self.doVar(strexpr);
            self.out.binCat = self.out.cat("channel");
            stderr.write("Will use category 'channel' to identify the %d channels\n" % self.out.binCat.numTypes())
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
                (pdf,coeff) = (self.getPdf(b,p), self.out.function("n_exp_bin%s_proc_%s" % (b,p)))
                extranorm = self.getExtraNorm(b,p)
                if extranorm:
                    self.doObj("n_exp_final_bin%s_proc_%s" % (b,p), "prod", "n_exp_bin%s_proc_%s, %s" % (b,p, extranorm))
                    coeff = self.out.function("n_exp_final_bin%s_proc_%s" % (b,p))
                pdfs.add(pdf); coeffs.add(coeff)
                if not self.DC.isSignal[p]:
                    bgpdfs.add(pdf); bgcoeffs.add(coeff)
            sum_s = ROOT.RooAddPdf("pdf_bin%s"       % b, "",   pdfs,   coeffs)
            sum_b = ROOT.RooAddPdf("pdf_bin%s_bonly" % b, "", bgpdfs, bgcoeffs)
            self.out._import(sum_s, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
            self.out._import(sum_b, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence())
    def doCombination(self):
        prefix = "modelObs" if len(self.DC.systs) else "model" # if no systematics, we build directly the model
        if len(self.DC.bins) > 1:
            for (postfixIn,postfixOut) in [ ("","_s"), ("_bonly","_b") ]:
                simPdf = ROOT.RooSimultaneous(prefix+postfixOut, prefix+postfixOut, self.out.binCat)
                for b in self.DC.bins:
                    simPdf.addPdf(self.out.pdf("pdf_bin%s%s" % (b,postfixIn)), b)
                self.out._import(simPdf, ROOT.RooFit.RecycleConflictNodes(), ROOT.RooFit.Silence(), ROOT.RooFit.Silence())
        else:
            self.out._import(self.out.pdf("pdf_bin%s"       % self.DC.bins[0]).clone(prefix+"_s"), ROOT.RooFit.Silence())
            self.out._import(self.out.pdf("pdf_bin%s_bonly" % self.DC.bins[0]).clone(prefix+"_b"), ROOT.RooFit.Silence())
        if len(self.DC.systs): # multiply by nuisances if needed
            self.doObj("model_s", "PROD", "modelObs_s, nuisancePdf")
            self.doObj("model_b", "PROD", "modelObs_b, nuisancePdf")

    ## --------------------------------------
    ## -------- High level helpers ----------
    ## --------------------------------------
    def prepareAllShapes(self):
        shapeTypes = []; shapeBins = []; shapeObs = {}
        for ib,b in enumerate(self.DC.bins):
            for p in ['data_obs']+self.DC.exp[b].keys():
                if len(self.DC.obs) == 0 and p == 'data_obs': continue
                if p != 'data_obs' and self.DC.exp[b][p] == 0: continue
                shape = self.getShape(b,p); norm = 0;
                if shape.ClassName().startswith("TH1"):
                    shapeTypes.append("TH1"); shapeBins.append(shape.GetNbinsX())
                    norm = shape.Integral()
                elif shape.InheritsFrom("RooDataHist"):
                    shapeTypes.append("RooDataHist"); 
                    shapeBins.append(shape.numEntries())
                    shapeObs[self.argSetToString(shape.get(0))] = shape.get(0)
                    norm = shape.sumEntries()
                elif shape.InheritsFrom("RooDataSet"):
                    shapeTypes.append("RooDataSet"); 
                    shapeObs[self.argSetToString(shape.get(0))] = shape.get(0)
                    norm = shape.sumEntries()
                elif shape.InheritsFrom("TTree"):
                    shapeTypes.append("TTree"); 
                elif shape.InheritsFrom("RooAbsPdf"):
                    shapeTypes.append("RooAbsPdf");
                else: raise RuntimeError, "Currently supporting only TH1s, RooDataHist and RooAbsPdfs"
                if norm != 0:
                    if p == "data_obs":
                        if len(self.DC.obs):
                            if abs(norm-self.DC.obs[b]) > 0.01:
                                raise RuntimeError, "Mismatch in normalizations for observed data in bin %s: text %f, shape %f" % (b,self.DC.obs[b],norm)
                    else:
                        if self.DC.exp[b][p] == -1: self.DC.exp[b][p] = norm
                        elif abs(norm-self.DC.exp[b][p]) > 0.01: 
                            raise RuntimeError, "Mismatch in normalizations for bin %s, process %d: rate %f, shape %f" % (b,p,self.DC.exp[b][p],norm)
        if shapeTypes.count("TH1") == len(shapeTypes):
            self.out.allTH1s = True
            self.out.mode    = "binned"
            self.out.maxbins = max(shapeBins)
            if self.options.verbose: stderr.write("Will use binning variable 'x' with %d bins\n" % self.out.maxbins)
            self.doVar("x[0,%d]" % self.out.maxbins); self.out.var("x").setBins(self.out.maxbins)
            self.out.binVar = self.out.var("x")
            self.out.binVars = ROOT.RooArgSet(self.out.binVar)
        else:
            if self.options.verbose: stderr.write("Will try to make a combined dataset\n")
            if self.options.verbose: stderr.write("Observables: %s\n" % str(shapeObs.keys()))
            if len(shapeObs.keys()) != 1:
                raise RuntimeError, "There's more than once choice of observables: %s\n" % str(shapeObs.keys())
            self.out.binVars = shapeObs.values()[0]
            self.out._import(self.out.binVars)
    def doCombinedDataset(self):
        if len(self.DC.bins) == 1:
            data = self.getData(self.DC.bins[0],'data_obs').Clone("data_obs")
            self.out._import(data)
            return
        if self.out.mode == "binned":
            combiner = ROOT.CombDataSetFactory(self.out.obs, self.out.binCat)
            for b in self.DC.bins: combiner.addSet(b, self.getData(b,"data_obs"))
            self.out.data_obs = combiner.done("data_obs","data_obs")
            self.out._import(self.out.data_obs)
        else: raise RuntimeException, "Only combined binned datasets are supported"

    ## -------------------------------------
    ## -------- Low level helpers ----------
    ## -------------------------------------
    def getShape(self,channel,process,syst="",_fileCache={},_neverDelete=[]):
        bentry = None
        if self.DC.shapeMap.has_key(channel): bentry = self.DC.shapeMap[channel]
        elif self.DC.shapeMap.has_key("*"):   bentry = self.DC.shapeMap["*"]
        else: raise KeyError, "Shape map has no entry for channel '%s'" % (channel)
        names = []
        if bentry.has_key(process): names = bentry[process]
        elif bentry.has_key("*"):   names = bentry["*"]
        else: raise KeyError, "Shape map has no entry for process '%s', channel '%s'" % (process,channel)
        if syst != "": names = [names[0], names[2]]
        else:          names = [names[0], names[1]]
        strmass = "%d" % self.options.mass if self.options.mass % 1 == 0 else str(self.options.mass)
        finalNames = [ x.replace("$PROCESS",process).replace("$CHANNEL",channel).replace("$SYSTEMATIC",syst).replace("$MASS",strmass) for x in names ]
        if not _fileCache.has_key(finalNames[0]): _fileCache[finalNames[0]] = ROOT.TFile.Open(finalNames[0])
        file = _fileCache[finalNames[0]]; objname = finalNames[1]
        if not file: raise RuntimeError, "Cannot open file %s (from pattern %s)" % (finalNames[0],names[0])
        if ":" in objname: # workspace:obj or ttree:xvar
            (wname, oname) = objname.split(":")
            wsp = file.Get(wname)
            if not wsp: raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (wname,finalNames[0],names[1],names[0])
            if wsp.ClassName() == "RooWorkspace":
                ret = wsp.data(oname)
                if ret: return ret;
                ret = wsp.pdf(oname)
                if ret: return ret;
                raise RuntimeError, "Object %s in workspace %s in file %s does not exist or it's neither a data nor a pdf" % (oname, wname, finalNames[0])
            elif wsp.ClassName() == "TTree":
                raise RuntimeError, "Another day"
            else:
                raise RuntimeError, "Object %s in file %s has unrecognized type %s" (wname, finalNames[0], wsp.ClassName())
        else: # histogram
            ret = file.Get(objname);
            if not ret: raise RuntimeError, "Failed to find %s in file %s (from pattern %s, %s)" % (objname,finalNames[0],names[1],names[0])
            if ret in _neverDelete: return ret
            ret.SetName("shape_%s_%s%s" % (process,channel, "_"+syst if syst else ""))
            if self.options.verbose: stderr.write("import (%s,%s) -> %s\n" % (finalNames[0],objname,ret.GetName()))
            _neverDelete.append(ret)
            return ret
    def getData(self,channel,process,syst=""):
        return self.shape2Data(self.getShape(channel,process,syst))
    def getPdf(self,channel,process,_cache={}):
        if _cache.has_key((channel,process)): return _cache[(channel,process)]
        shapeNominal = self.getShape(channel,process)
        nominalPdf = self.shape2Pdf(shapeNominal)
        morphs = []
        for (syst,pdf,args,errline) in self.DC.systs:
            if pdf != "shape": continue
            if errline[channel][process] != 0:
                shapeUp   = self.getShape(channel,process,syst+"Up")
                shapeDown = self.getShape(channel,process,syst+"Down")
                if shapeUp.ClassName()   != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                if shapeDown.ClassName() != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                morphs.append((syst,errline[channel][process],self.shape2Pdf(shapeUp),self.shape2Pdf(shapeDown)))
        if len(morphs) == 0: return nominalPdf
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
        _cache[(channel,process)] = ROOT.VerticalInterpPdf("shape_%s_%s_morph" % (channel,process), "", pdfs, coeffs, minscale, 0)
        return _cache[(channel,process)]
    def getExtraNorm(self,channel,process):
        terms = []
        shapeNominal = self.getShape(channel,process)
        if shapeNominal.InheritsFrom("RooAbsPdf"): 
            # no extra norm for parametric pdfs (could be changed)
            return None
        normNominal = 0
        if shapeNominal.InheritsFrom("TH1"): normNominal = shapeNominal.Integral()
        elif shapeNominal.InheritsFrom("RooDataHist"): normNominal = shapeNominal.sumEntries()
        else: return None    
        for (syst,pdf,args,errline) in self.DC.systs:
            if pdf != "shape": continue
            if errline[channel][process] != 0:
                shapeUp   = self.getShape(channel,process,syst+"Up")
                shapeDown = self.getShape(channel,process,syst+"Down")
                if shapeUp.ClassName()   != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                if shapeDown.ClassName() != shapeNominal.ClassName(): raise RuntimeError, "Mismatched shape types for channel %s, process %s, syst" % (channel,process,syst)
                kappaUp,kappaDown = 1,1
                if shapeNominal.InheritsFrom("TH1"):
                    kappaUp,kappaDown = shapeUp.Integral(),shapeDown.Integral()
                elif shapeNominal.InheritsFrom("RooDataHist"):
                    kappaUp,kappaDown = shapeUp.sumEntries(),shapeDown.sumEntries()
                kappaUp /=normNominal; kappaDown /= normNominal
                if abs(kappaUp-1) < 1e-3 and abs(kappaDown-1) < 1e-3: continue
                # if errline[channel][process] == <x> it means the gaussian should be scaled by <x> before doing pow
                # for convenience, we scale the kappas
                kappasScaled = [ pow(x, errline[channel][process]) for x in kappaDown,kappaUp ]
                terms.append("AsymPow(%f,%f,%s)" % (kappasScaled[0], kappasScaled[1], syst))
        return ",".join(terms) if terms else None;
    def shape2Data(self,shape,_cache={}):
        if not _cache.has_key(shape.GetName()):
            if shape.ClassName().startswith("TH1"):
                rebinh1 = ROOT.TH1F(shape.GetName()+"_rebin", "", self.out.maxbins, 0.0, float(self.out.maxbins))
                for i in range(1,min(shape.GetNbinsX(),self.out.maxbins)+1): 
                    rebinh1.SetBinContent(i, shape.GetBinContent(i))
                rdh = ROOT.RooDataHist(shape.GetName(), shape.GetName(), ROOT.RooArgList(self.out.var("x")), rebinh1)
                self.out._import(rdh)
                _cache[shape.GetName()] = rdh
            elif shape.ClassName() in ["RooDataHist", "RooDataSet"]:
                return shape
            else: raise RuntimeError, "shape2Data not implemented for %s" % shape.ClassName()
        return _cache[shape.GetName()]
    def shape2Pdf(self,shape,_cache={}):
        if not _cache.has_key(shape.GetName()+"Pdf"):
            if shape.ClassName().startswith("TH1"):
                rdh = self.shape2Data(shape)
                rhp = self.doObj("%sPdf" % shape.GetName(), "HistPdf", "{x}, %s" % shape.GetName())
                _cache[shape.GetName()+"Pdf"] = rhp
            elif shape.InheritsFrom("RooAbsPdf"):
                _cache[shape.GetName()+"Pdf"] = shape
            elif shape.InheritsFrom("RooDataHist"):
                rhp = ROOT.RooHistPdf("%sPdf" % shape.GetName(), "", self.out.binVars, shape) 
                self.out._import(rhp)
                _cache[shape.GetName()+"Pdf"] = rhp
            else: 
                raise RuntimeError, "shape2Data not implemented for %s (%s)" % (shape.ClassName())
        return _cache[shape.GetName()+"Pdf"]
    def argSetToString(self,argset):
        names = []
        it = argset.createIterator()
        while True:
            arg = it.Next()
            if not arg: break
            names.append(arg.GetName())
        return ",".join(names)

