from ROOT import *
import sys
import os
import subprocess

#List of search terms from command line
searchTerms = []

#Read in directory and open root file
file = TFile(sys.argv[1],"UPDATE")
#Take in multiple arguments.
for i in range(2,len(sys.argv)):
    if "_" in sys.argv[i]:
        searchTerms.append(sys.argv[i])
if len(searchTerms) == 0:
    print "No valid request for overlaid plotting given"
    print "Remember to include a '_' separated search term, with at least one '_'"
#plotsToSum = sys.argv[2]
#Initialise auto-y-scaling. This will become more complicated with generalisation.
for plotsKey in searchTerms:
    plotsToDraw = []
    yMax  = -1000000.0
    yMin = 1000000.0
    scaling = false
    for key in file.GetListOfKeys():
        if key.GetClassName() != "TH1F":continue
        #Add efficiencies to arrays - eventually these will become command line arguments
        parse = true
        for part in plotsKey.split("_"):
            if part not in key.GetName():
                parse = false
                break
        if not parse: continue
#        if plotsKey in key.GetName():
        plotsToDraw.append(key.GetName())
        #Attempt to extract yMax and yMin from the configuration file
        cmsBase = subprocess.Popen('echo $CMSSW_BASE',shell=True, stdout=subprocess.PIPE).communicate()[0][:-1]
        temp= subprocess.Popen('grep ' + key.GetName() + " " + cmsBase + '/src/DQM/SiStripHistoricInfoClient/test/NewHDQM/cfg/*',shell=True, stdout=subprocess.PIPE).communicate()[0].split(':')[0]
        f = open(temp,'r')
        for line in f:
            if key.GetName() in line:
                for line in f:
                    if '[' in line:
                        break
                    if 'yMin' in line.split('=')[0]:
                        scaling = true
                        if float(line.split('=')[1]) < float(yMin): yMin = float(line.split('=')[1])
                    if 'yMax' in line.split('=')[0]:
                        scaling = true
                        if float(line.split('=')[1]) > float(yMax): yMax = float(line.split('=')[1])
    #If there's no information in the config, set it to the max and min of the plots themselves.
    if not scaling:
        for b in plotsToDraw:
            tkey = file.GetListOfKeys().FindObject(b)
            hist = tkey.ReadObj()
            if hist.GetMaximum() > yMax : yMax = hist.GetMaximum()
            if hist.GetMinimum() < yMin : yMin = hist.GetMinimum()
        yDif = yMax - yMin
        yMax = yMax + 0.1*yDif
        yMin = yMin - 0.1*yDif
    #Making room for the legend
    yDif = yMax - yMin
    yMax = yMax + 0.5*yDif

    if len(plotsToDraw) < 1: continue
    print "Overlaying the following plots: ", plotsToDraw
    #Plot barrel efficiencies
    C=TCanvas("C","C",600,600)
    C.cd()
    for b in plotsToDraw:
        tkey = file.GetListOfKeys().FindObject(b)
        hist = tkey.ReadObj()
        #Automatically scale y-axis to combining plots
        hist.SetMarkerColor(plotsToDraw.index(b)+1)
        hist.SetMarkerStyle(20+plotsToDraw.index(b))
        hist.SetMarkerSize(1.0 + ( plotsToDraw.index(b) > 1 ) * 0.2 )
        binSF = int(float(hist.GetNbinsX())/30.0)+1
        #save axis so that it can be reverted after drawing
        xaxis = hist.GetXaxis()
        if binSF > 1:
            for i in range(1,hist.GetNbinsX()+1):
                if i%(binSF)!=0: hist.GetXaxis().SetBinLabel(i,"")
        #Plots the histograms together - must plot the first one without "same"
        if plotsToDraw.index(b) == 0 :
            hist.Draw("p")
            hist.GetYaxis().SetRangeUser(yMin,yMax)
            hist.GetXaxis().SetTitle("")
            hist.GetYaxis().SetTitle("") 
        else: hist.Draw("p same")
        for i in range(1,hist.GetNbinsX()+1):hist.GetXaxis().SetBinLabel(i,xaxis.GetBinLabel(i)) #I feel this is somewhat redundant since the root file still won't include this extra information, though it probably should. TODO.
    #Temporarily change binning for ease of looking
    C.BuildLegend()
    C.GetListOfPrimitives().FindObject(plotsToDraw[0]).SetTitle(plotsKey.replace("_"," "))    
    C.SetTitle(plotsKey)
    C.SetName(plotsKey)
    C.SaveAs(os.path.dirname(os.path.abspath(sys.argv[1])) + "/" + plotsKey + ".png")
    file.cd()
    C.Write()
    #C.SaveAs(os.path.dirname(os.path.abspath(sys.argv[1])) + "/" + plotsKey + ".root")
    C.Close()
file.Close()
