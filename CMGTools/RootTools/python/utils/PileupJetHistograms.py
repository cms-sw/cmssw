from CMGTools.RootTools.statistics.Histograms import Histograms
import CMGTools.External.External
from ROOT import TH1F, TH2F, TFile, THStack, TF1, TGraphErrors, TPrincipal, TObjString, TObject, PileupJetIdAlgo

from bisect import bisect

# --------------------------------------------------------------------------------------
def mkBinLabels(bins, scale=1., fmt="%1.0f", addOFlow=True):
    labels = []
    last = bins[0]
    for bin in bins[1:]:
        labels.append( str(fmt+"_"+fmt) % (last*scale,bin*scale) )
        last=bin
    if addOFlow:
        labels.append( str(fmt+"_inf") % (last*scale) )
    return labels

# --------------------------------------------------------------------------------------
def findBin(bins,val):
    return bisect(bins,val)-1

# --------------------------------------------------------------------------------------
def formatTitle(title,args):
    unit = args["unit"]
    if unit == "":
        args["unitx"]=""
        args["unity"]="%(perbin)s"
    else:
        args["unitx"]="("+unit+")"
        args["unity"]="%(perbinunit)s "+unit
    args["jetbin"] = "%(vtxbin)s%(ptbin)s%(etabin)s" % args
    return title % args    
        
# --------------------------------------------------------------------------------------
def fillTitle(h):
    binw = h.GetBinWidth(1)
    if binw == 1:
        perbin = "/ bin"
        perbinunit = "/"
    else:
        perbin = "/ %1.2g" % binw
        perbinunit = perbin
    h.SetTitle( h.GetTitle() % { "perbin" : perbin, "perbinunit" : perbinunit } )
    h.GetXaxis().SetTitle( h.GetXaxis().GetTitle() % { "perbin" : perbin, "perbinunit" : perbinunit } )
    h.GetYaxis().SetTitle( h.GetYaxis().GetTitle() % { "perbin" : perbin, "perbinunit" : perbinunit } )
        

# --------------------------------------------------------------------------------------
class PileupJetHistograms(Histograms) :
    """
    
    """
    ## protoypes for histogram booking 
    prototypes={
        
	"mva"        : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,-1.,1.),
        
	"jetPt"      : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c",300,0,150),
	"jetEta"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),
	"jetPhi"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),
        "jetM"       : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c^{c}",100,0,50),

    	"nCharged"   : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,50),
	"nNeutrals"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,50),
	"nParticles" : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,50),
        
	"chgEMfrac"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neuEMfrac"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"chgHadrfrac": ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neuHadrfrac": ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	
	"d0"         : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","cm",100,0,2),
	"dZ"         : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","cm",100,0,10),

	"leadPt"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c",70,0,35),
	"leadEta"    : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),
	"leadPhi"    : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),
	"secondPt"   : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s""GeV/c",70,0,35),
	### "secondEta"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),      
	### "secondPhi"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),
	"leadNeutPt" : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c",70,0,35),
	### "leadNeutEta": ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),      
	### "leadNeutPhi": ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),
	"leadEmPt"   : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c",70,0,35),
	### "leadEmEta"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),
	### "leadEmPhi"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),
	"leadChPt"   : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s","GeV/c",70,0,35),
	### "leadChEta"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",101,-5.05,5.05,),
	### "leadChPhi"  : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",33,-3.21,3.21),

	"leadFrac"       : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"secondFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"thirdFrac"      : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"fourthFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"leadChFrac"       : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"secondChFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"thirdChFrac"      : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"fourthChFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
        
	"leadNeutFrac"       : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"secondNeutFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"thirdNeutFrac"      : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"fourthNeutFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"leadEmFrac"       : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"secondEmFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"thirdEmFrac"      : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"fourthEmFrac"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"dRLeadCent" :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"dRLead2nd"  :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"dRMean"     :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"dRMeanNeut" :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"dRMeanEm"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"dRMeanCh"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),

	"etaW"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"phiW"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"majW"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),
	"minW"   :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",50,0,0.5),

	"frac01"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"frac02"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"frac03"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"frac04"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"frac05"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"chFrac01"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"chFrac02"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"chFrac03"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"chFrac04"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"chFrac05"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	
	"neutFrac01"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neutFrac02"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neutFrac03"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neutFrac04"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"neutFrac05"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"emFrac01"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"emFrac02"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"emFrac03"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"emFrac04"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
	"emFrac05"     : ("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),

	"ptD"        :("%(name)s %(hname)s%(jetbin)s;%(hname)s %(unitx)s;Jets %(unity)s",100,0,1.),
        }

    # --------------------------------------------------------------------------------------
    def __init__(self,name,vtxbins=None,ptbins=None,etabins=None,vtxlabels=None,ptlabels=None,etalabels=None,reweight=None,
                 jetIdMva=None):
        """
        """
        from ROOT import PileupJetIdentifier
        ## deal with nvtx/pt/eta binning and labels
        self.name = name
        self.vtxbins = vtxbins
        self.ptbins = ptbins
        self.etabins = etabins
        self.reweight = reweight
        
        if self.reweight:
            if len(self.reweight) == 2:
                self.do_reweight = getattr(self.reweight[1],"Eval")
            else:
                self.do_reweight = lambda x : self.reweight[2][findBin(self.reweight[1],x)]

        if self.vtxbins and not vtxlabels:
            self.vtxlabels = [ "_vtx%s" % l for l in  mkBinLabels(self.vtxbins) ]
        elif vtxlabels:
            self.vtxlabels = vtxlabels
        else:
            self.vtxlabels = [""]

        if self.ptbins and not ptlabels:
            self.ptlabels = [ "_pt%s" % l for l in mkBinLabels(self.ptbins) ]
        elif ptlabels:
            self.ptlabels = ptlabels
        else:
            self.ptlabels = [""]
            
        if self.etabins and not etalabels:
            self.etalabels = [ "_eta%s" % l for l in  mkBinLabels(self.etabins) ]
        elif etalabels:
            self.etalabels = etalabels
        else:
            self.etalabels = [""]

        ## book histograms and keep track of what needs to be filled
        self.fillers = []
        if jetIdMva:
            print jetIdMva
            self.identifier = PileupJetIdAlgo(*jetIdMva)
            self.jetIdMva = jetIdMva
            self.runMva = True
        else:
            self.identifier = PileupJetIdAlgo()
            self.jetIdMva = ()
            self.runMva = True
        
        vNames = ""
        for name,proto in self.prototypes.iteritems():
            self.fillers.append( (self.bookHistos(name,proto), name) )
            if name != "mva":
                vNames += ":%s" % name

        ### ## book covariance matrixes
        ### self.principals = tuple( tuple( tuple( ("principal_%s%s%s%s" % (self.name,eta,vtx,pt), TPrincipal(len(self.prototypes)-1, "") ) 
        ###                                        for eta in self.etalabels ) for pt in self.ptlabels ) for vtx in self.vtxlabels)
        ## call the Histograms constructor to get everything registered 
        super (PileupJetHistograms,self).__init__ (self.name)
        
        ### ## add covariance matrixes to list of objects to write
        ### for a in self.principals:
        ###     for b in a:
        ###         for name,principal in b:
        ###             principal.SetName(name)
        ###             self.hists.append(principal)
        ### 
        ### ## hack to override Write method for TObjString
        ### self.vNames = TObjString(vNames)
        ### self.vNames.name = "vNames"
        ### self.vNames.Write = lambda : TObject.Write(self.vNames,"vNames")
        ### self.hists.append(self.vNames)

        
    # --------------------------------------------------------------------------------------
    def bookHistos(self,hname,proto):
        title = proto[0]
        if type(proto[1]) == list:
            aux = proto[1]
            args = proto[2:]
        elif type(proto[1]) == str:
            aux = {"unit":proto[1]}
            args = proto[2:]
        else:
            aux = { "unit" : "" }
            args = proto[1:]
        t = tuple( tuple(
            tuple( TH1F("%s%s%s%s_%s" % (self.name,eta,vtx,pt,hname),
                        formatTitle(title, dict({"name":self.name, "hname":hname, "vtxbin": vtx, "ptbin":pt, "etabin":eta}.items()+aux.items())),*args)
                    for eta in self.etalabels ) for pt in self.ptlabels ) for vtx in self.vtxlabels) 
        ## self.__setattr__("list_%s" % hname) = t
        self.addHistos(t)
        
        return t

    # --------------------------------------------------------------------------------------
    def addHisto(self,h):
        fillTitle(h)
        self.__setattr__(h.GetName().replace("%s_"%self.name,""),h)

    # --------------------------------------------------------------------------------------
    def addHistos(self,histos):
        if type(histos[0]) == tuple or type(histos[0]) == list:
            for hi in histos:
                for h in hi: self.addHistos(h)
        else:
            for h in histos: self.addHisto(h)

    # --------------------------------------------------------------------------------------
    def fillJet(self,jet,vertexes):
        from array import array
        ptbin = 0
        etabin = 0
        if self.ptbins:  ptbin  = findBin(self.ptbins,jet.pt())
        if self.etabins: etabin = findBin(self.etabins,abs(jet.eta()))
        if self.vtxbins: vtxbin = findBin(self.vtxbins,len(vertexes))

        if ptbin < 0 or etabin < 0 or vtxbin < 0:
            return

        w = 1.
        if self.reweight:
            w = self.do_reweight(getattr(jet,self.reweight[0])())
        
        try:
            puid = jet.puIdentifier
        except:
            puidalgo = self.identifier
            try:
                jet.puIdentifier = puidalgo.computeIdVariables(jet.sourcePtr().get(),0.,vertexes[0],self.runMva)
            except:
                jet.puIdentifier = puidalgo.computeIdVariables(jet,0.,vertexes[0],self.runMva)
            puid = jet.puIdentifier
            ### jet.puIdentifier = PileupJetIdentifier(puid)
        
        ### a = array('d')
        for t,m in self.fillers:
            v = getattr(puid,m)()
            ### if m != "mva":
            ###     a.append(v)
            t[vtxbin][ptbin][etabin].Fill( v, w )
        ### self.principals[vtxbin][ptbin][etabin][1].AddRow(a)

    # --------------------------------------------------------------------------------------
    def summary(self):
        ### for a in self.principals:
        ###     for b in a:
        ###         for name,principal in b:
        ###             print principal.GetName()
        ###             principal.Print("MSE")
        ###             principal.GetCovarianceMatrix().Print("MSE")
        pass
        
    # --------------------------------------------------------------------------------------
    def fillEvent(self,event,vertices):
        for jet in event:
            self.fillJet(jet,vertices)
