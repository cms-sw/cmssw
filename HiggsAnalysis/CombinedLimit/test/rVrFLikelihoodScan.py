#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
import array
from optparse import OptionParser

# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *

parser = OptionParser(usage="usage: %prog [options] rvrf.root -o output \nrun with --help to get list of options")
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:strictSMLikeHiggs",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("-o", "--out",      dest="out",   default="rVrFLikelihoodScan.root",  type="string", help="output file (if none, it will print to stdout). Required for binary mode.")
parser.add_option("-v", "--verbose",  dest="verbose",  default=0,  type="int",      help="Verbosity level (0 = quiet, 1 = verbose, 2+ = more)")
parser.add_option("-m", "--mass",     dest="mass",     default=125.7,  type="float",  help="Higgs mass to use. Will also be written in the Workspace as RooRealVar 'MH'.")
parser.add_option("-d" ,"--decays",   dest="decays",   default=None,  type="string", help="decays to include (comma separated, if unspecified use all)")
parser.add_option("--algo",      dest="algo",   default="grid",  type="string", help="algorithm (only grid supported for now)")
parser.add_option("-p", "--poi", dest="poi", default=[],  type="string", action="append", help="POI to process")
parser.add_option("--setPhysicsModelParameters",      dest="setPOIVals",   default=[],  type="string", action="append", help="POI values to set (poi=value,poi2=value2,...)")
parser.add_option("--setPhysicsModelParameterRanges", dest="setPOIRanges", default=[],  type="string", action="append", help="POI values to set (poi=min,max:poi2=min2,max2:...)")
parser.add_option("--floatOtherPOI", dest="floatOtherPOI",  default=1,  type="int", help="Float other POI")
parser.add_option("--points", dest="points",  default=0,  type="int", help="Points to scan")

(options, args) = parser.parse_args()

## set up some dummy options for the sake of the ModelBuilder
options.bin = True
options.fileName = args[0]
options.cexpr = False
## and create a model builder
DC = Datacard()
MB = ModelBuilder(DC, options)

## Load physics model
(physModMod, physModName) = options.physModel.split(":")
__import__(physModMod)
mod = modules[physModMod]
physics = getattr(mod, physModName)
if mod     == None: raise RuntimeError, "Physics model module %s not found" % physModMod
if physics == None or not isinstance(physics, PhysicsModel): 
    raise RuntimeError, "Physics model %s in module %s not found, or not inheriting from PhysicsModel" % (physModName, physModMod)
physics.setPhysicsOptions(options.physOpt)
## Attach model to tools, and declare parameters
MB.setPhysics(physics)
MB.physics.doParametersOfInterest()

## Define all possible decay modes, and the dominant V, F productions
decays = [ 
  ('hbb', ('VH', 'ttH')),
  ('htt', ('qqH', 'ggH')),
  ('hgg', ('qqH', 'ggH')),
  ('hww', ('qqH', 'ggH')),
  ('hzz', ('qqH', 'ggH')),
]

## Open input file with histograms
input = ROOT.TFile.Open(args[0])
## Create the multichannel likelihood
likelihood = ROOT.rVrFLikelihood("NLL","NLL")
## Create RooRealVars for dummy scale factors, in case they're needed
MB.doVar("__zero__[0]")
MB.doVar("__one__[1]")
for decay,(prodv,prodf) in decays:
    if options.decays and decay not in options.decays: continue
    ## get scaling factors
    muv = MB.physics.getHiggsSignalYieldScale(prodv,decay,'8TeV')
    muf = MB.physics.getHiggsSignalYieldScale(prodf,decay,'8TeV')
    ## if necessary convert 0 and 1 to names of RooAbsReals
    if muv == 0: muv = "__zero__"
    if muv == 1: muv = "__one__"
    if muf == 0: muf = "__zero__"
    if muf == 1: muf = "__one__"
    ## get histogram of 2*deltaNLL
    hist = input.Get("rvrf_scan_2d_%s_th2" % decay)
    ## add as channel in the likelihood
    likelihood.addChannel(hist, MB.out.function(muv), MB.out.function(muf))

## finalize POIs and set mass
MB.physics.done()
MB.out.var("MH").setVal(options.mass)

## import in workspace (maybe not needed?)
MB.out._import(likelihood)

if options.verbose > 1: MB.out.Print("V")

## Initialize output tree
output =  ROOT.TFile.Open(options.out, "RECREATE")
tree = ROOT.TTree("limit","limit")
# copy dummy structure of the ones from combine
limit = array.array('d',[0.]); tree.Branch("limit", limit, "limit/D");
limitErr = array.array('d',[0.]); tree.Branch("limitErr", limitErr, "limitErr/D");
mh = array.array('d',[options.mass]); tree.Branch("mh", mh, "mh/D");
itoy = array.array('i',[0]); tree.Branch("iToy", itoy, "iToy/I");
quantileExpected = array.array('f',[0.]); tree.Branch("quantileExpected", quantileExpected, "quantileExpected/F");
## this is the only variable that is really useful of the standard ones
deltaNLL = array.array('f',[0.]); tree.Branch("deltaNLL", deltaNLL, "deltaNLL/F");

## Set parameter ranges and values from command line
for pranges in options.setPOIRanges:
    for prange in pranges.split(":"):
        (pname,prange) = prange.split("=")
        lo,hi = prange.split(",")
        MB.out.var(pname).setRange(float(lo),float(hi))
for pvals in options.setPOIVals:
    for pset in pvals.split(","):
        (pname,pval) = pset.split("=")
        MB.out.var(pname).setVal(float(pval))

## Do global fit
fullMinimizer = ROOT.RooMinimizer(likelihood)
fullMinimizer.setPrintLevel(-1)
fullMinimizer.setStrategy(2)
fullMinimizer.minimize("Minuit","minimize")
fullMinimizer.minos() ## minos is cheap on these models ;-)

## Print it out
print "Best fit point: "
fullMinimizer.save().Print("V")

## select parameters to scan
poiList = ROOT.RooArgList(); 
if options.poi != []:
    if not options.floatOtherPOI:
        ## if needed, freeze all the parameters that are not of interest
        poiList.add(MB.out.set("POI"))
        for i in xrange(poiList.getSize()):
            if poiList.at(i).GetName() not in options.poi:
                poiList.at(i).setConstant(True)
        poiList = ROOT.RooArgList()
    else:
        poiList.add(MB.out.set("POI"))
        for i in xrange(poiList.getSize()):
            if poiList.at(i).GetName() not in options.poi:
                print "Will profile ",poiList.at(i).GetName()
        poiList = ROOT.RooArgList()
    for pn in options.poi:
        ## make list of parameters
        p = MB.out.var(pn)
        poiList.add(p)
        ## set them constant so that we can profile the others
        p.setConstant(True)
else:
    ## take all parameters
    poiList.add(MB.out.set("POI"))

## Allocate variables and create tree branches 
poi = [ array.array('f',[poiList.at(i).getVal()]) for i in xrange(poiList.getSize()) ]
for i in xrange(poiList.getSize()):
    poiNam = poiList.at(i).GetName()
    tree.Branch(poiNam, poi[i], poiNam+"/F")    


## Add global minimum to tree
nll0 = likelihood.getVal()
for i in xrange(poiList.getSize()): 
    poi[i][0] = poiList.at(i).getVal()
deltaNLL[0] = 0.;
tree.Fill()

# Now prepare minimizer for profiling
constrMinimizer = ROOT.RooMinimizer(likelihood)
constrMinimizer.setPrintLevel(-1)
constrMinimizer.setStrategy(2)
# and save if we should minimize or not
mustMinim = (poiList.getSize() != MB.out.set("POI").getSize() and options.floatOtherPOI)

if options.algo == "grid":
    ## set default number of points if needed
    if options.points == 0: options.points = int(pow(200,len(poi)))
    ## ger parameter ranges
    pmin = [ poiList.at(i).getMin() for i in xrange(len(poi)) ]
    pmax = [ poiList.at(i).getMax() for i in xrange(len(poi)) ]
    if len(poi) == 1:
        print "1D scan of %s with %d points" % (poiList.at(0).GetName(), options.points)
        for i in xrange(options.points):
            x = pmin[0] + (i+0.5)*(pmax[0]-pmin[0])/options.points
            poiList.at(0).setVal(x); poi[0][0] = x
            if mustMinim and likelihood.getVal() < 999: constrMinimizer.minimize("Minuit","minimize")
            deltaNLL[0] = likelihood.getVal() - nll0;
            tree.Fill()
    elif len(poi) == 2:
        print "2D scan of %s, %s with %d points" % (poiList.at(0).GetName(), poiList.at(1).GetName(), options.points)
        sqrn = int(ceil(sqrt(float(options.points))))
        deltaX = (pmax[0]-pmin[0])/sqrn
        deltaY = (pmax[1]-pmin[1])/sqrn
        for i in xrange(sqrn):
            for j in xrange(sqrn):
                x = pmin[0] + (i+0.5)*deltaX;
                y = pmin[1] + (j+0.5)*deltaY;
                poiList.at(0).setVal(x); poi[0][0] = x
                poiList.at(1).setVal(y); poi[1][0] = y
                #MB.out.allVars().Print("V")
                if mustMinim and likelihood.getVal() < 999: constrMinimizer.minimize("Minuit","minimize")
                deltaNLL[0] = likelihood.getVal() - nll0;
                tree.Fill()
elif options.algo not in [ "none", "singles" ]:
    raise RuntimeError, "Unknown algorithm: '%s'" % options.algo

# save tree to disk
tree.Write() 
