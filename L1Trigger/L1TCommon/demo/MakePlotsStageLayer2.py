import os,sys
##from ROOT import *  # global import, easy but NOT recommended
from ROOT import gROOT, gDirectory, gPad, gSystem, gStyle
from ROOT import TFile, TTree, TH1D, TCanvas

inputFileName = "demo_output.root"
treeName = "Events"
prefix = "l1t"
suffixHW = "BXVector_Layer2HW__L1TEMULATION.obj.data_.l1t::L1Candidate."
suffixPH = "BXVector_Layer2Phys__L1TEMULATION.obj.data_.l1t::L1Candidate."


channels = ["EGamma", "Jet", "EtSum", "Tau"]
plotVars = ["hwPt", "hwEta", "hwPhi", "pt", "eta", "phi"]


#############################################
def setHistStyle(hist, xtitle):
    hist.GetXaxis().SetTitle(xtitle)
    hist.GetXaxis().SetNdivisions(518)
    hist.GetXaxis().SetLabelSize(0.05)
    hist.GetYaxis().SetLabelSize(0.05)
    
#############################################
def drawHist(tree, suffix, object, var, nBins, min, max, cuts):
    h = TH1D(var, "",  int(nBins), min, max)
    setHistStyle(h, var)
    drawStr = prefix + object + suffix + var
    if object=="EtSum": drawStr = drawStr + "_[0]>>"
    else: drawStr = drawStr +  "_>>"
    
    tree.Draw( drawStr  + var, cuts, "goff")
    h.SetMinimum(0.5)    
    h.SetMaximum( 5 * h.GetMaximum() )
    
    canvas= TCanvas()
    h.Draw()
    canvas.SetLogy()
    canvas.Update()
    canvas.SaveAs( object + "_" + var + ".png")
    canvas.WaitPrimitive()


#############################################
file = TFile(inputFileName)
tree = file.Get(treeName)

gStyle.SetOptStat("emr")
gStyle.SetHistLineWidth(2)
gStyle.SetCanvasDefH(500)
gStyle.SetCanvasDefW(500)
gStyle.SetTitleXSize(0.05)
gStyle.SetTitleXOffset(0.85)


for i in range( len(channels) ):
    object = channels[i]
    for j in range( len(plotVars) ):
        var    = plotVars[j]
        max    = 0
        min    = 0
        bins   = 0

        ######## hwPt binning ####################
        if (object=="Jet" or object=="EtSum" or
            object=="Tau") and var=="hwPt":
            max = pow(2, 10) 
            min = 0
            bins = (max - min)/8
        elif object=="EGamma" and var=="hwPt":
            max = pow(2, 6) - 0.5
            min = -0.5
            bins = max - min
            
        ######## hwEta binning ####################
        if var=="hwEta":
            max = 20.5
            min = -0.5
            bins = 21
            
        ######## hwPhi binning ####################
        if var=="hwPhi":
            max = 17.5
            min = -0.5
            bins = 18

        ######## pt binning ####################
        if (object=="Jet" or object=="EtSum" or
            object=="Tau") and var=="pt":
            max = 1000
            min = 0
            bins = 100
        elif object=="EGamma" and var=="pt":
            max = 32
            min = 0
            bins = 16

        ######## eta binning ####################
        if (object=="Jet" or object=="EtSum" or
            object=="Tau") and var=="eta":
            max = 5.05
            min = -5.05
            bins = 101
        elif object=="EGamma" and var=="eta":
            max = 3.05
            min = -3.05
            bins = 61            
            
        ######## phi binning ####################
        if var=="phi":
            max = 3.25
            min = -3.25
            bins = 65

            
        howmany = 4
        if object=="EtSum": howmany = 1
        cuts = "BranchListIndexes<" + str(howmany)


        suffix = ""
        if "hw" in var: suffix = suffixHW
        elif var=="pt" or var=="eta" or var=="phi": suffix = suffixPH
        
        drawHist( tree, suffix, object, var, bins, min, max, cuts) 

