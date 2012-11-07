###############################################
# makeFCcontour.py
# runs over output of HybridNew toys to extract
# either 1D or 2D Feldman-Cousins contours

# Original Author - Nicholas Wardle - IC London
# Run with ./makeFCcontour files [options]
###############################################

#!/usr/bin/env python

from array import array
import sys
import ROOT
ROOT.gROOT.SetBatch(1)

# Dummy variables for getting tree values (not used with HybridNew output)
dumx = array('f',[0.])
dumy = array('f',[0.])
dumv = array('f',[0.])
dumd = array('i',[0])

errLevel = 10E-8 
defaultYval = 1.

class physicsPoint:
  
  def __init__(self,x):
    self.label = "x=%.2f, y=%.2f"%(x[0],x[1])
    self.x = x[0]
    self.y = x[1]
    self.data = -9999.
    self.toys = []
    self.hasdata = False

    self.nToysPass = 0 
    self.nToys     = 0

  def is_point(self,valx,valy):
    if 	abs(valx-self.x)>errLevel: return False 
    if  abs(valy-self.y)>errLevel: return False
    return True

  def get_n_toys(self):
    return self.nToys
  
  def commit_toy(self,val):
    if self.hasdata: 
        if val>self.data : self.nToysPass+=1
        self.nToys+=1
    else : self.toys.append(val)

  def set_data(self,dat):

    if self.hasdata: return
    self.data = dat
    self.hasdata = True
    
    for v in self.toys: 
        if v>self.data: self.nToysPass+=1
    self.nToys+=len(self.toys)
    self.toys = [] # Clean up 

  def has_data(self):
    return self.hasdata

  def get_cl(self):
    return float(self.nToys-self.nToysPass)/self.nToys

  def isInsideContour(self,cl):
    nToysPass = float(self.nToysPass)
    return int((nToysPass/self.nToys>=(1-cl)))


def findStringValue(fullstr,substr):
  si = fullstr.find(substr)
  if si==-1 :sys.exit("No toy results found for variable %s"%substr)
  ei = fullstr.find("_",si+len(substr))
  v  = float(fullstr[si+len(substr):ei])
  return v


def getPoints(tree,varx,vary):
  points = set([])
  gPoints = []

  # grab all the distributions:
  hypoResults = tree.GetListOfKeys()
  hypoResults = filter(lambda ke: "HypoTestResult" in ke.GetName(),hypoResults)

  #for a in range(tree.GetEntries()):
  for k_i,key in enumerate(hypoResults):
    keyname = key.GetName()
    if not ("HypoTestResult" in keyname): continue
    result = key.ReadObj()
    dumx   = findStringValue(keyname,varx)
    if vary=="": dumy=defaultYval # dummy value
    else:dumy   = findStringValue(keyname,vary) 

    if (options.xrange[0]>-999. and options.xrange[1]>-999.) and (dumx>options.xrange[1]+errLevel or dumx<options.xrange[0]-errLevel): continue
    if not vary=="":
      if (options.yrange[0]>-999. and options.yrange[1]>-999.) and (dumy>options.yrange[1]+errLevel or dumy<options.yrange[0]-errLevel): continue
    
    pointExists = False 

    dataval = result.GetTestStatisticData()
    toys    = result.GetAltDistribution().GetSamplingDistribution()

    for gp in gPoints:
      if gp.is_point(dumx,dumy):
          pointExists=True
          if not gp.has_data(): gp.set_data(dataval)
	  for tv in toys: 
		gp.commit_toy(tv)
          break

    if not pointExists: 
      newPoint = physicsPoint([dumx,dumy])
      newPoint.set_data(dataval)
      for tv in toys: 
		newPoint.commit_toy(tv)
      gPoints.append(newPoint)
	

  return gPoints


def findRange(mySet,index):
  
  if index == 0 : myList = [m.x for m in mySet]
  else :myList = [m.y for m in mySet]
  myIndexSet  = set(myList)
  myFinalList = list(myIndexSet)
  return min(myFinalList),max(myFinalList),len(myFinalList)

def returnPoints(mySet,index):
  if index == 0 : myList = [m.x for m in mySet]
  else :myList = [m.y for m in mySet]
  newList = list(set(myList))
  newList.sort()
  extrapoint = 2*newList[-1]-newList[-2]
  return newList+[extrapoint]
  
exampleDone = False
points   = []
def get_confs(option, opt_str, value, parser):
  setattr(parser.values, option.dest, value.split(','))

def mergePoints(original,appended):
  # take 2 lists of physics points and merge the second into the first
  for p in appended:

      pointexists=False
      for po in original:
          if po.is_point(p.x,p.y):
               pointexists=True

               if p.has_data(): 
                 po.nToys+=p.nToys
                 po.nToysPass+=p.nToysPass
                 po.set_data(p.data) # in po already has data, this wont do anything
               else: 
                  for toy in p.toys: po.commit_toy(toy) # this will just do the right thing
               break

      if not pointexists:
        original.append(p)

def findInterval(pts,cl):
	# start by walking along the variable and check if crosses a CL point
	crossbound = [ pt[1]<=cl for pt in pts ]
	rcrossbound = crossbound[:]
	rcrossbound.reverse()

	minci = 0
	maxci = len(crossbound)-1
	min = pts[0][0]
	max = pts[maxci][0]

	for c_i,c in enumerate(crossbound): 
		if c : 
			minci=c_i
			break
	
	for c_i,c in enumerate(rcrossbound): 
		if c : 
			maxci=len(rcrossbound)-c_i-1
			break

	if minci>0: 
		y0,x0 = pts[minci-1][0],pts[minci-1][1]
		y1,x1 = pts[minci][0],pts[minci][1]
		min = y0+((cl-x0)*y1 - (cl-x0)*y0)/(x1-x0)
		
	if maxci<len(crossbound)-1: 
		y0,x0 = pts[maxci][0],pts[maxci][1]
		y1,x1 = pts[maxci+1][0],pts[maxci+1][1]
		max = y0+((cl-x0)*y1 - (cl-x0)*y0)/(x1-x0)

	return min,max
	
# Main Routine  ---------------------------------------------------------//
ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptStat(0)

from optparse import OptionParser
parser = OptionParser()
parser = OptionParser(usage="usage: %prog [options] files (or list of files) \nrun with --help to get list of options")
parser.add_option("","--cl",dest="cl",default="",type='str',action='callback',callback=get_confs,help="Set Confidence Levels (comma separated) (eg 0.68 for 68%)")
parser.add_option("-x","--xvar",dest="xvar",default="",type='str',help="Name of branch for x-value")
parser.add_option("-y","--yvar",dest="yvar",default="",type='str',help="Name of branch for y-value")
parser.add_option("","--xrange",dest="xrange",default=(-9999.,-9999.),nargs=2,type='float',help="only pick points inside here")
parser.add_option("","--yrange",dest="yrange",default=(-9999.,-9999.),nargs=2,type='float',help="only pick points inside here")
parser.add_option("","--d1",dest="oned",default=False,action="store_true",help="Run 1D FC (ie just report confidence belt). In this case --yvar is irrelevant")
parser.add_option("-o","--out",dest="out",default="plots2DFC.root",type='str',help="Output File for 2D histos/1D confidence scan")
parser.add_option("-t","--tdir",dest="treename",default='toys',type=str,help="Name of TDirectory for toys inside grid files")

(options,args)=parser.parse_args()
allFiles = args[:]

if options.cl :confidenceLevels = [float(c) for c in options.cl]
else: confidenceLevels=[]

if options.yvar and options.oned: sys.exit("For 1D intervals, only specify x variable")

# 2D set of points, labelled x,y
xvar = options.xvar
yvar = options.yvar

if options.oned: print "Calculating 1D FC interval for ", xvar
else : print "Constructing 2D FC contours, x=",xvar, "y=",yvar
treeName  = str(options.treename)
print "Grabbing all points from files (Getting Points can be very slow if all contained in one file!)"

for fileName in allFiles:
  print "Opening File -- ", fileName

  tFile = ROOT.TFile.Open(fileName)
  tToys = tFile.Get(treeName)

  if options.oned: cpoints = getPoints(tToys,xvar,"")
  else: cpoints = getPoints(tToys,xvar,yvar)
  mergePoints(points,cpoints)
  tFile.Close()

outFile = ROOT.TFile(options.out ,"RECREATE")

# For 1D /************************************************************************/
if options.oned:

  tgrX = ROOT.TGraph(); tgrX.SetMarkerStyle(21); tgrX.SetMarkerSize(1.0)
  # can sort the points since only 1D
  points = sorted(points,key=lambda pt:pt.x)
  values = [(pt.x,pt.get_cl()) for pt in points]
  # make a graph too
  for pt_i,pt in enumerate(values):
    xval = pt[0]
    zval = pt[1]
    tgrX.SetPoint(pt_i,zval,xval)
    
  for c_i,confLevel in enumerate(confidenceLevels):
    lowcl,highcl = findInterval(values,confLevel)
    print "%f < %s < %f"%(lowcl,xvar,highcl) ,"(%.3f CL)"%confLevel

  outFile.cd(); 
  tgrX.Write("confcurve");

# For 2D /************************************************************************/
# One histogram with ALL values of CL  

else:
  # Now have every point on the grid, make a TH2
  xmin,xmax,nx = findRange(points,0)
  ymin,ymax,ny = findRange(points,1)

  xbins = returnPoints(points,0)
  ybins = returnPoints(points,1)
  xbins_d = array('f',xbins)
  ybins_d = array('f',ybins)
  tgrXY  = ROOT.TGraph2D()
  ntXY   = ROOT.TGraph2D()

  for pt_i,pt in enumerate(points):
    xval = pt.x
    yval = pt.y
    zval = pt.get_cl()
    print "Found %d toys for point (%f,%f)" % (pt.get_n_toys(),xval,yval)
    tgrXY.SetPoint(pt_i,xval,yval,zval)
    ntXY.SetPoint(pt_i,xval,yval,pt.get_n_toys())

  print "Found points in grid: %s = %.2f->%.2f (%d points), %s = %.2f->%.2f (%d points) "\
      %(xvar,xmin,xmax,nx,yvar,ymin,ymax,ny)

  clXY = tgrXY.GetHistogram()
  clXY.GetXaxis().SetTitle(xvar)
  clXY.GetYaxis().SetTitle(yvar)
  clXY.SetName("h2_cl")

  nthistXY = ntXY.GetHistogram()
  nthistXY.GetXaxis().SetTitle(xvar)
  nthistXY.GetYaxis().SetTitle(yvar)
  nthistXY.SetName("n_toys")

  outFile.cd(); clXY.Write(); nthistXY.Write();

  # Histogram for each contour plot, going through and checking if point passes CL is 
  # more honest but looks uglier (good to check agreement in anycase)

  for c_i,confLevel in enumerate(confidenceLevels):
    print "Finding %.3f confidence region"%(confLevel)
    grXY = ROOT.TH2F("h2_confcontour_%d"%(100*confLevel),\
	";%s;%s"%(xvar,yvar),len(xbins)-1,xbins_d,len(ybins)-1,ybins_d)
    for pt_i,pt in enumerate(points):
      xval = pt.x
      yval = pt.y
      zval = pt.isInsideContour(confLevel)
      bin = grXY.FindBin(xval,yval)
      grXY.SetBinContent(bin,zval)

    grXY.SetContour(2); grXY.SetLineWidth(2);grXY.Draw()
    grXY.GetXaxis().SetTitle(xvar)
    grXY.GetYaxis().SetTitle(yvar)
    outFile.cd(); grXY.Write()

outFile.Close()
print "Created File ",outFile.GetName(), " containing confidance contours"
# ---------------------------------------------------------------------//



