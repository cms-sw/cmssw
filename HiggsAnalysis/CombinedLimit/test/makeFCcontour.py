#!/usr/bin/env python

###############################################
# makeFCcontour.py
# runs over output of HybridNew toys to extract
# either 1D or 2D Feldman-Cousins contours

# Original Author - Nicholas Wardle - IC London
# Run with ./makeFCcontour files [options]
###############################################


from array import array
import os,sys
import ROOT
ROOT.gROOT.SetBatch(1)

# Used later
gDataNLL = 0

def get_confs(option, opt_str, value, parser):
  setattr(parser.values, option.dest, value.split(','))

# can recursively search for files containing toys
def return_filenames(rootdir,searchstring) :
  fileslist = []
  for folders,subs,files in os.walk(rootdir):
      for fi in files:
	  fullpath = os.path.join(folders,fi)
          if ".root" in fi: fileslist.append(fullpath)
  if searchstring: fileslist=filter(lambda x: searchstring in x,fileslist)
  return fileslist

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] files (or list of files) \nrun with --help to get list of options")
parser.add_option("","--cl",dest="cl",type='str',action='callback',callback=get_confs,help="Set Confidence Levels (comma separated) (eg 0.68 for 68%)")
parser.add_option("","--overlap",dest="overlap",default=-0.99,type='float',help="Define a circular region of overlap for points in grid to be merged")
parser.add_option("-x","--xvar",dest="xvar",default="",type='str',help="Name of branch for x-value")
parser.add_option("-y","--yvar",dest="yvar",default="",type='str',help="Name of branch for y-value")
parser.add_option("","--xrange",dest="xrange",default=(-9999.,-9999.),nargs=2,type='float',help="only pick points inside here")
parser.add_option("","--yrange",dest="yrange",default=(-9999.,-9999.),nargs=2,type='float',help="only pick points inside here")
parser.add_option("","--d1",dest="oned",default=False,action="store_true",help="Run 1D FC (ie just report confidence belt). In this case --yvar is irrelevant")
parser.add_option("-o","--out",dest="out",default="plots2DFC.root",type='str',help="Output File for 2D histos/1D confidence scan")
parser.add_option("-t","--tdir",dest="treename",default='toys',type=str,help="Name of TDirectory for toys inside grid files")
parser.add_option("","--teststat",dest="teststat",default='PL',type=str,help="Test statistic used (chooses 1 or 2 sided)")
parser.add_option("-d","--dataNLL",dest="datafile",default='',type=str,help="Input data (ie NLL scan) to use instead of from jobs")
parser.add_option("","--minToys",dest="minToys",default='-10',type=int,help="Minimum number of toys to accept a point")
parser.add_option("","--storeToys",dest="storeToys",default=False,action="store_true",help="Keep histograms of the llr for toys (and the datavalue) in the output file (warning, increases run time)")
#parser.add_option("-f","--filesdir",dest="filesdir",default='',type=str,help="Directory to recursively search for toys, use dir:reg to search fo regular expression inside dir")
parser.add_option("-f","--filesdir",dest="filesdir",type='str',action='callback',callback=get_confs,help="Directory to recursively search for toys, use dir:reg to search fo regular expression inside dir (comma separate for multiple dirs)")
(options,args)=parser.parse_args()


# Dummy variables for getting tree values (not used with HybridNew output)
dumx = array('f',[0.])
dumy = array('f',[0.])
dumv = array('f',[0.])
dumd = array('i',[0])

errLevel = 10E-6 
defaultYval = 1.

cutGrid  = False
cutGridy = False

class physicsPoint:
  
  def __init__(self,x):
    self.label = "x=%.2f, y=%.2f"%(x[0],x[1])
    self.name  = "x=%.2f_y=%.2f"%(x[0],x[1])
    self.x = x[0]
    self.y = x[1]
    self.data = -9999.
    self.toys = []
    self.hasdata = False
    self.savetoys = False
    self.do_overlap = False
    self.overlap_region = 0.

    self.nToysPass = 0 
    self.nToys     = 0
    self.onesided  = True
    self.usedatanll= False
    #self.datanllgr = 0

  def get_data(self):
   return self.data

  def set_data_nll(self,gr):
   
   #self.datanllgr=gr.Clone()
   self.usedatanll=True
   self.data=gr.Eval(self.x)
   self.hasdata=True

  def set_twosided(self):
   self.onesided=False	
 
  def set_overlap(self, val):
    self.do_overlap = True
    self.overlap_region = val

  def save_toys(self):
    self.savetoys=True

  def is_point(self,valx,valy):
    if  self.do_overlap:
	circlesize =  (valx-self.x)**2 + (valy-self.y)**2
	if abs(ROOT.TMath.Sqrt(circlesize)) < self.overlap_region:
		return True
    if 	abs(valx-self.x)>errLevel: return False 
    if  abs(valy-self.y)>errLevel: return False
    return True

  def get_n_toys(self):
    return self.nToys
  
  def commit_toy(self,val):
     if abs(val)>10000:return # something i'm sure went strange if so
     if self.onesided and val<0: return
     #if abs(val)<0.0001: val = 0.0001
     if self.hasdata: 
        if val>=self.data : self.nToysPass+=1
        self.nToys+=1
        if self.savetoys: self.toys.append(val)
     else : self.toys.append(val)

  def set_data(self,dat):

    if self.hasdata and self.data > -999.: return
   # if self.usedatanll:
#	self.data=self.datanllgr.Eval(self.x)
#	self.hasdata=True
#	return

    self.data = dat
    self.hasdata = True
    
    for v in self.toys:
        if v>=self.data: self.nToysPass+=1
    # Clean up unless we want to keep toys
    if (not self.savetoys): 
        self.nToys+=len(self.toys)
	self.toys = []  

  def has_data(self):
    return self.hasdata

  def get_cl(self):
    
    if self.nToys==0: return 1
    if self.onesided:
      nominalTail = float(self.nToysPass)/self.nToys
      #nominalTail+= ROOT.TMath.Prob(8,1) #### HACK!
      #if 2*self.data>1: return 1-(ROOT.Math.chisquared_cdf_c(2*self.data,1))
      #else:  return 1-nominalTail
      return 1-nominalTail
    else :
      rtail = float(self.nToysPass)/self.nToys
      if rtail < .5: return (1.-0.5-rtail)
      else: return 2*(rtail-0.5)

  def get_cl_err(self):
    #return 0
    if self.nToys==0: return 1
    e = self.get_cl()
    N = self.get_n_toys()
    k = e*N
    return (1./N)*(k*(1-k/N))**0.5

  def isInsideContour(self,cl):
    nToysPass = float(self.nToysPass)
    if self.onesided:
      return int((float(nToysPass)/self.nToys>=(1-cl)))
    else:
      rtail = float(nToysPass)/self.nToys
      ltail = 1.-rtail

      return min([ltail,rtail])>(cl/2)

  def histogramToys(self):
    min=0
    if not self.onesided: min=-10
    htoys = ROOT.TH1F("hToys_%s"%self.name,"hToys_%s"%self.name,100,min,10)
    hdata = ROOT.TH1F("hData_%s"%self.name,"hData_%s"%self.name,1000,min,10)
    for ty in self.toys: htoys.Fill(ty)
    if self.has_data(): hdata.Fill(self.data)

    return htoys.Clone(),hdata.Clone()

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

    if ( cutGrid ) and (dumx>options.xrange[1]+errLevel or dumx<options.xrange[0]-errLevel): continue
    if not vary=="":
      if (cutGridy ) and (dumy>options.yrange[1]+errLevel or dumy<options.yrange[0]-errLevel): continue
    
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
      if options.storeToys: newPoint.save_toys()
      if options.overlap>0: newPoint.set_overlap(options.overlap)
      if options.teststat=="TwoSided" or options.teststat=="LEP": newPoint.set_twosided()
      if options.datafile: newPoint.set_data_nll(gDataNLL)
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

def mergePoints(original,appended):
  # take 2 lists of physics points and merge the second into the first
  for p in appended:

      pointexists=False
      for po in original:
          if po.is_point(p.x,p.y):
               pointexists=True

               if p.has_data() and not options.storeToys: 
                 po.nToys+=p.nToys
                 po.nToysPass+=p.nToysPass
                 po.set_data(p.data) # if po already has data, this wont do anything
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

allFiles = args[:]
if len(allFiles)==0 and (not options.filesdir): 
	parser.print_usage()
	sys.exit()

# options.filesdir could have two parts:
if options.filesdir:
 for direxp in options.filesdir:
  searchdir = ""
  searchexp = ""

  if ":" in direxp:
   searchdir,searchexp = direxp.split(":")
  else : 
	searchdir = direxp

  allFiles+=return_filenames(searchdir,searchexp)

if options.cl :confidenceLevels = [float(c) for c in options.cl]
else: confidenceLevels=[0.68]

if options.yvar and options.oned: sys.exit("For 1D intervals, only specify x variable")

cutGrid = options.xrange[0]>-999. and options.xrange[1]>-999.
if not options.oned :cutGridy = (options.yrange[0]>-999. and options.yrange[1]>-999.)

# 2D set of points, labelled x,y
xvar = options.xvar
yvar = options.yvar

# Optional use of NLL from pre-performed scan 
if options.datafile and options.oned:
  tFileNLL=ROOT.TFile(options.datafile)
  # Draw the graph 
  nlltree=tFileNLL.Get("limit")
  nlltree.Draw("deltaNLL:%s"%xvar)
  gDataNLL = (ROOT.gPad.FindObject("Graph").Clone())

if options.oned: print "Calculating 1D FC interval for ", xvar
else : print "Constructing 2D FC contours, x=",xvar, "y=",yvar
if options.teststat=="TwoSided" or options.teststat=="LEP" : sys.exit("Using two sided test statistic NOT Implemented fully yet! ")
treeName  = str(options.treename)
print "Grabbing all points from files (Getting Points can be very slow if all contained in one file!)"


n_tot_files = len(allFiles)
failedFiles = []
for f_it,fileName in enumerate(allFiles):
  print "Opening File (%d/%d) -- "%(f_it,n_tot_files), fileName

  tFile = ROOT.TFile.Open(fileName)
  if tFile == None : 
	print "File Corrupted, skipping"
	failedFiles.append(fileName)
	continue
  tToys = tFile.Get(treeName)
  if tToys == None : 
	print "File doesn't contain ", treeName
	failedFiles.append(fileName)
	continue

  if options.oned: cpoints = getPoints(tToys,xvar,"")
  else: cpoints = getPoints(tToys,xvar,yvar)
  mergePoints(points,cpoints)
  tFile.Close()

outFile = ROOT.TFile(options.out ,"RECREATE")
  
# Do this first in 1D case otherwise points is overwritten
if options.storeToys: 
    for pt in points:
	hty,hdt = pt.histogramToys() 
	outFile.cd(); hty.Write(); hdt.Write()

# For 1D /************************************************************************/
if options.oned:

  tgrX = ROOT.TGraphErrors(); tgrX.SetMarkerStyle(21); tgrX.SetMarkerSize(1.0)
  tgrD = ROOT.TGraph(); tgrD.SetMarkerStyle(21);tgrD.SetMarkerSize(1.0)
  # can sort the points since only 1D
  points = sorted(points,key=lambda pt:pt.x)
  values  = [(pt.x,pt.get_cl(),pt.get_cl_err(),pt.get_data()) for pt in points]
  numtoys = [pt.get_n_toys() for pt in points]
  # make a graph too
  grCounter=0
  for pt_i,pt in enumerate(values):
    xval = pt[0]
    zval = pt[1]
    eval = pt[2]
    dval = pt[3]
    print "Found %d toys for point (%f)" % (numtoys[pt_i],xval), "CL = " ,zval
    if options.minToys > -1  and numtoys[pt_i] < options.minToys: 
	print " -----> (not enough toys, ignore point) "
    else:
    	tgrD.SetPoint(grCounter,xval,dval)
    	tgrX.SetPoint(grCounter,zval,xval)
    	tgrX.SetPointError(grCounter,eval,0)
	grCounter+=1
  
    
  for c_i,confLevel in enumerate(confidenceLevels):
    lowcl,highcl = findInterval(values,confLevel)
    print "%f < %s < %f"%(lowcl,xvar,highcl) ,"(%.3f CL)"%confLevel

  outFile.cd(); 
  tgrX.Write("confcurve");
  tgrD.Write("dataNLL");

# For 2D /************************************************************************/
# One histogram with ALL values of CL, also one TGraph per value of CL (although 
# they won't look great probably as the interpolation is wrong. It will say whehter
# a point is in or out of each CL contour
# Also a histogram with number of toys thrown (found) at each point
# Also one TGraph with each stored point in it 

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
  ptXY   = ROOT.TGraph()

  pt_i = 0
  for i,pt in enumerate(points):
    xval = pt.x
    yval = pt.y
    ptXY.SetPoint(pt_i,xval,yval)
    zval = pt.get_cl()
    print "Found %d toys for point (%f,%f)" % (pt.get_n_toys(),xval,yval)
    if options.minToys > -1  and pt.get_n_toys() < options.minToys: 
	print " -----> (not enough toys, ignore point) "
    else:
   	 tgrXY.SetPoint(pt_i,xval,yval,zval)
    	 ntXY.SetPoint(pt_i,xval,yval,pt.get_n_toys())
    	 pt_i+=1

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

  ptXY.GetXaxis().SetTitle(xvar)
  ptXY.GetYaxis().SetTitle(yvar)
  ptXY.SetName("points")

  outFile.cd(); clXY.Write(); nthistXY.Write(); ptXY.Write()

  # Histogram for each contour plot, going through and checking if point passes CL is 
  # more honest but looks uglier (good to check agreement in anycase)

  for c_i,confLevel in enumerate(confidenceLevels):
    print "Finding %.3f confidence region"%(confLevel)
    #grXY = ROOT.TH2F("h2_confcontour_%d"%(100*confLevel),\
    #";%s;%s"%(xvar,yvar),len(xbins)-1,xbins_d,len(ybins)-1,ybins_d)
    grXY = ROOT.TGraph2D()
    grXY.SetName("h2_confcontour_%d"%(100*confLevel))
    for pt_i,pt in enumerate(points):
      xval = pt.x
      yval = pt.y
      zval = pt.isInsideContour(confLevel)
      #bin = grXY.FindBin(xval,yval)
      grXY.SetPoint(pt_i,xval,yval,zval)

    #grXY.SetContour(2); 
    grXY.SetMarkerSize(0.8);grXY.Draw()
    grXY.GetXaxis().SetTitle(xvar)
    grXY.GetYaxis().SetTitle(yvar)
    outFile.cd(); grXY.Write()

outFile.Close()
print "Created File ",outFile.GetName(), " containing confidance contours"
if options.storeToys: print "Saved histograms of toys and data to file"
#for f in failedFiles : print f, # for debugging/removing failed files
# ---------------------------------------------------------------------//



