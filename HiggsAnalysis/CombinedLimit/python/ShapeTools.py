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
        stderr.write("qui si parra' la tua nobilitate\n")
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
                (pdf,coeff) = (self.getPdf(p,b), self.out.function("n_exp_bin%s_proc_%s" % (b,p)))
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
                shape = self.getShape(p,b); norm = 0;
                if shape.ClassName().startswith("TH1"):
                    shapeTypes.append("TH1"); shapeBins.append(shape.GetNbinsX())
                    norm = shape.Integral()
                elif shape.InheritsFrom("RooDataHist"):
                    shapeTypes.append("RooDataHist"); 
                    shapeBins.append(shape.numEntries())
                    shapeObs[self.argSetToString(shape.get(0))] = shape.get(0)
                elif shape.InheritsFrom("RooDataSet"):
                    shapeTypes.append("RooDataSet"); 
                    shapeObs[self.argSetToString(shape.get(0))] = shape.get(0)
                elif shape.InheritsFrom("TTree"):
                    shapeTypes.append("TTree"); 
                elif shape.InheritsFrom("RooAbsPdf"):
                    shapeTypes.append("RooAbsPdf");
                else: raise RuntimeError, "Currently supporting only TH1s, RooDataHist and RooAbsPdfs"
                if p != 'data_obs' and norm != 0:
                    if self.DC.exp[b][p] == -1: self.DC.exp[b][p] = norm
                    elif abs(norm-self.DC.exp[b][p]) > 0.01: 
                        stderr.write("Mismatch in normalizations for bin %s, process %d: rate %f, shape %f" % (b,p,self.DC.exp[b][p],norm))
        if shapeTypes.count("TH1") == len(shapeTypes):
            self.out.allTH1s = True
            self.out.mode    = "binned"
            self.out.maxbins = max(shapeBins)
            stderr.write("Will use binning variable 'x' with %d bins\n" % self.out.maxbins)
            self.doVar("x[0,%d]" % self.out.maxbins); self.out.var("x").setBins(self.out.maxbins)
            self.out.binVar = self.out.var("x")
            self.out.binVars = ROOT.RooArgSet(self.out.binVar)
        else:
            stderr.write("Will try to make a combined dataset\n")
            stderr.write("Observables: %s\n" % str(shapeObs.keys()))
            if len(shapeObs.keys()) != 1:
                raise RuntimeError, "There's more than once choice of observables: %s\n" % str(shapeObs.keys())
            self.out.binVars = shapeObs.values()[0]
            self.out._import(self.out.binVars)
    def doCombinedDataset(self):
        if len(self.DC.bins) == 1:
            data = self.getData('data_obs',self.DC.bins[0]).Clone("data_obs")
            self.out._import(data)
            return
        if self.out.mode == "binned":
            combiner = ROOT.CombDataSetFactory(self.out.obs, self.out.binCat)
            for b in self.DC.bins: combiner.addSet(b, self.getData("data_obs",b))
            self.out.data_obs = combiner.done("data_obs","data_obs")
            self.out._import(self.out.data_obs)
        else: raise RuntimeException, "Only combined binned datasets are supported"

    ## -------------------------------------
    ## -------- Low level helpers ----------
    ## -------------------------------------
    def getShape(self,process,channel,syst="",_fileCache={},_neverDelete=[]):
        pentry = None
        if self.DC.shapeMap.has_key(process): pentry = self.DC.shapeMap[process]
        elif self.DC.shapeMap.has_key("*"):   pentry = self.DC.shapeMap["*"]
        else: raise KeyError, "Shape map has no entry for process '%s'" % (process)
        names = []
        if pentry.has_key(channel): names = pentry[channel]
        elif pentry.has_key("*"):   names = pentry["*"]
        else: raise KeyError, "Shape map has no entry for process '%s', channel '%s'" % (process,channel)
        if syst != "": names = [names[0], names[2]]
        else:          names = [names[0], names[1]]
        finalNames = [ x.replace("$PROCESS",process).replace("$CHANNEL",channel).replace("$SYSTEMATIC",syst) for x in names ]
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
            ret.SetName("shape_%s_%s%s" % (process,channel, "_"+syst if syst else ""))
            stderr.write("import (%s,%s) -> %s\n" % (finalNames[0],objname,ret.GetName()))
            _neverDelete.append(ret)
            return ret
    def getData(self,process,channel,syst=""):
        return self.shape2Data(self.getShape(process,channel,syst))
    def getPdf(self,process,channel,syst=""):
        return self.shape2Pdf(self.getShape(process,channel,syst))
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

