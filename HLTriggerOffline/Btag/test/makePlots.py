#!/usr/bin/env python

######################### file options ##################################
DQMfileOld="../test/DQM_V0001_R000000001__CMSSW_test__RelVal__TrigVal.root"
DQMfileNew="../test/DQM_V0001_R000000001__CMSSW_test__RelVal__TrigVal.root"
labelNew = "New"
labelOld = "Old"

########################## load libraries #################################

import os, string, re, sys, math

try:
		import ROOT
except:
		print "\nCannot load PYROOT, make sure you have setup ROOT in the path"
		print "and pyroot library is also defined in the variable PYTHONPATH, try:\n"
		if (os.getenv("PYTHONPATH")):
			print " setenv PYTHONPATH ${PYTHONPATH}:$ROOTSYS/lib\n"
		else:
			print " setenv PYTHONPATH $ROOTSYS/lib\n"
		sys.exit()

folder="plots"
try:
	os.mkdir(folder)
except:
	print "folder " + folder + " already existing"

from ROOT import TFile
from ROOT import TCanvas
from ROOT import TLegend
#from ROOT import SetOwnership
#from ROOT import THStack
#from ROOT import TLatex
#from ROOT import TH1
from ROOT import TH1F
from ROOT import TGraphErrors
#from ROOT import TVectorD
#from ROOT

########################## define a function that return plots given a TFile #################################

def GetPlots(_file0):
#	_file0=TFile(filename)
	dir1 = _file0.Get("DQMData")
	dir2 = dir1.Get("Run 1")
	dir3 = dir2.Get("HLT")
	dir4 = dir3.Get("Run summary")
	plots=[]
	for type in dir4.GetListOfKeys():
		dirType= dir4.Get(type.GetName())
		for triggerKey in dirType.GetListOfKeys():
			triggerDir=dirType.Get(triggerKey.GetName())
			for plotKey in triggerDir.GetListOfKeys():
				plotPointer=triggerDir.Get(plotKey.GetName())
				plot=plotPointer
				if(plot.GetName()=="efficiency"): 
					for plotEfficiencyKey in plotPointer.GetListOfKeys():
						plot=plotPointer.Get(plotEfficiencyKey.GetName())
						plots=plots+[plot.Clone(triggerKey.GetName() + plot.GetName())]
				else:
					plots=plots+[plot.Clone(triggerKey.GetName() + plot.GetName())]
	
	return plots

########################## read DQM plots #################################
fileNew=TFile(DQMfileOld)
plotsNew = GetPlots(fileNew)
fileOld=TFile(DQMfileNew)
plotsOld = GetPlots(fileOld)

##### for kind of plots save a .png superimposing the New with the Old #####

ROOT.gROOT.SetBatch()
ROOT.gStyle.SetOptStat(0)
c1 = TCanvas("c1","",1280,720)
c1.SetGridx()
c1.SetGridy()
legend = TLegend(0.07,0.85,0.2,0.93);

first=True
for plotNew in plotsNew:
	for plotOld in plotsOld:
		if(plotNew.GetName()==plotOld.GetName()):
			plotOld.SetLineColor(4)
			plotOld.SetMarkerColor(4)
#			plotOld.SetFillColor(4)
			plotNew.SetLineColor(2)
			plotNew.SetMarkerColor(2)
#			plotNew.SetFillColor(2)
#			plotNew.SetFillStyle(3002)
			
			plotNew.SetLineWidth(2)
			plotOld.SetLineWidth(2)
			if first:
				legend.AddEntry(plotNew,labelNew,"l");
				legend.AddEntry(plotOld,labelOld,"l");
			
			if plotNew.GetName().rfind("mistagrate)")>0:
				plotOld.SetMinimum(0.001)
				plotNew.SetMinimum(0.001)
				c1.SetLogy(1)
			else:
				c1.SetLogy(0)
			
			plotOld.Draw()
			plotNew.Draw("same")
			legend.Draw()
			c1.SaveAs(folder+"/"+plotNew.GetName()+".png")
			first=False
			
