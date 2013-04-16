import ROOT
ROOT.gROOT.ProcessLine(".L $CMSSW_BASE/lib/$SCRAM_ARCH/libHiggsAnalysisCombinedLimit.so");
ROOT.gROOT.SetBatch(1)
#ROOT.gStyle.SetNumberContours(255)

import sys,numpy,array
import itertools

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] file \nrun with --help to get list of options")
parser.add_option("-M","--Model",dest="model",default="CvCf",type='str',help="Name of model used in model builder")
parser.add_option("-m","--mh",dest="mh",default=125.8,type='float',help="Lightest Higgs Mass")
(options,args)=parser.parse_args()

# Can do full matrix of prod*decay
energies = ["7TeV"]
production_channels 	= ["ggH","qqH","WH","ZH","VH","ttH"]
decay_channels 		= ["hgg","hzz","hww","htt","hbb"]
decay_modes 		= ["hgg","hvv","hff"]

# Stepping of parameters in the model 
step = 0.05

# Coloured plots
def set_palette(ncontours=999):

    # default palette, looks cool
    stops = [0.00, 0.34, 0.61, 0.84, 1.00]
    red   = [0.00, 0.00, 0.87, 1.00, 0.51]
    green = [0.00, 0.81, 1.00, 0.20, 0.00]
    blue  = [0.51, 1.00, 0.12, 0.00, 0.00]

    s = array.array('d', stops)
    r = array.array('d', red)
    g = array.array('d', green)
    b = array.array('d', blue)

    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)


# Function which add a Double_t branch for each entry in a list
def createBranches(tree, params):
    for p in params.keys():
	tree.Branch(p,params[p],"%s/Double_t"%(p))


def fillGrid(func,graph,txtfile,tree,c_vals):

	point = 0
	for val in c_vals:

	  # set the parameters 
	  it = params.createIterator()
	  for j in range(nparams):
		p = it.Next()
		if p==None : break		
		p.setVal(val[j])
		parameter_vals[p.GetName()][0]=val[j]
		txtfile.write("%1.2f   "%(val[j]))

	  mu = func.getVal()

	  if nparams == 2 : 
	 	graph.SetPoint(point,val[0],val[1],mu)
		point+=1

	  txtfile.write("%2.4f\n"%(mu))
	  parameter_vals["mu"][0]=mu
	  tree.Fill()
	  point += 1

def fillOutputs(func,graph,txtfile,tree):

	full_grid = []

	# create a range for each parameter	
	it = params.createIterator()
	for j in range(nparams):
		p = it.Next()
		vals = numpy.arange(p.getMin(),p.getMax()+step,step)
		full_grid.append(vals)
	
	c_vals = itertools.product(*full_grid)

        fillGrid(func,graph,txtfile,tree,c_vals)

	return 0

def makePlot(name,tgraph):

	c = ROOT.TCanvas("c","c",600,600)

	it = params.createIterator()
	xparam = it.Next()
	yparam = it.Next()

	minxval = xparam.getMin()
	maxxval = xparam.getMax()
	minyval = yparam.getMin()
	maxyval = yparam.getMax()

	thist = tgraph.GetHistogram()

	thist.SetTitle(name)
	thist.GetXaxis().SetTitle(xparam.GetName())
	thist.GetYaxis().SetTitle(yparam.GetName())
	#thist.GetZaxis().SetTitle("#sigma(%s)*BR(%s)/#sigma_{SM}"%(prod,decay))
	
	thist.GetXaxis().SetRangeUser(minxval,maxxval)
	thist.GetYaxis().SetRangeUser(minyval,maxyval)

	# Contour for mu=1 (no scaling applied)
	thist.Draw("COLZ")
	thistContour = thist.Clone()
	thistContour.SetContour(2)
	thistContour.SetContourLevel(1,1)
	thistContour.SetLineColor(10)
	thistContour.Draw("CONT3same")
	
	# lines for the SM values
	
	lvert = ROOT.TLine(1,minyval,1,maxyval)
	lhorz = ROOT.TLine(minxval,1,maxxval,1)
	lvert.SetLineStyle(2)
	lhorz.SetLineStyle(2)
	lvert.Draw()
	lhorz.Draw()
	
	c.SaveAs("%s.pdf"%(name))
	#c.SetLogz()
	#c.SaveAs("%s_logscale.pdf"%(name))

def produceScan(modelname,extname,proddecay,work):

	# Get the appropriate mapping 
	name = "%s_%s_%s"%(modelname,extname,proddecay)
	func = work.function(name)
	if not func: return 

	# Produce a plot of the scaling parameters
	tgraph = ROOT.TGraph2D()

	# And a .txt file of the numbers:
	txtfile = open("%s_%s.txt"%(modelname,proddecay),"w")
	txtfile.write("%s - %s scaling factors\n"%(modelname, proddecay))
	
	it = params.createIterator()
	for j in range(nparams):
		p = it.Next()
		if p==None : break		
		txtfile.write("%s    "%(p.GetName()))
	txtfile.write("mu\n")
	
	# And a TTree 
	tr = ROOT.TTree("%s"%(proddecay),"%s"%(proddecay))
	createBranches(tr,parameter_vals)

	# This is the loop over points in the model 		
	fillOutputs(func,tgraph,txtfile,tr)

	# make a plot 
	if nparams == 2: makePlot("%s_%s"%(modelname,proddecay),tgraph)
	else: print "Skipping plots (nparams != 2)"
	
	# Write The Tree:
	tr.Write()

	# close the txtfile
	txtfile.close()

# MAIN FUNCTION 

wsfile = args[0]

# Select a mass for the (light) Higgs boson 
mHval = options.mh

tfile = ROOT.TFile.Open(wsfile)
work  = tfile.Get("w")

mH = work.var("MH")
mH.setVal(mHval)

# model config defines the parameters of interest
mc_s   = work.genobj("ModelConfig")
params  = mc_s.GetParametersOfInterest()
nparams = params.getSize()
print "Number of parameters in Model: ", nparams
params.Print()

parameter_vals	= {"mu":array.array('d',[0])}

# create a dictionary object which containts name, empty array and ref to parm
iter = params.createIterator()
while 1: 
  p = iter.Next()
  if p == None: break
  name = p.GetName()
  parameter_vals[name] = array.array('d',[0])

#for p in params: print p.GetName()
# Output file for ROOT TTrees
tree_output = ROOT.TFile("%s_trees.root"%options.model,"RECREATE")

set_palette(ncontours=255) # For colored plots

# Simple production x-section and branching ratios
for decay in decay_modes:

	ext = "BRscal"
	produceScan(options.model,ext,decay,work)

# Full Matrix sigma*BR available in Model:
#for e in energies:
for prod in production_channels:
   for decay in decay_channels:

	ext = "XSBRscal"
	proddecay = "%s_%s"%(prod,decay)
	produceScan(options.model,ext,proddecay,work)

	

tree_output.Close()

