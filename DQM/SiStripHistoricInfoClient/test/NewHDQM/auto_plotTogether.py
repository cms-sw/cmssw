from ROOT import *
import re
import os
import string
import sys
f = {}
h = {}
can = {}
lop = {}
mgo = {}
log = {}
g   = {}
obs = {}
#to get all the plots we want to overlay
plotlist = {
    1: "NClust_OFFTk_T*count*",
    2: "NClust_OFFTk_T*mean*",
    3: "NClust_ONTk_T*count*",
    4: "NClust_ONTk_T*mean*",
    5: "adc_ONTk_*",
    6: "centralCharge_OFFTk_*mpv*",
    7: "centralStoN_OFFTk_T*mean*",
    8: "centralStoN_OFFTk_T*mpv*",
    9: "centralStoN_ONTk_T*mean*",
    10: "centralStoN_ONTk_T*mpv*",
    11: "charge_ONTk_*",
    12: "clustCharge_OFFTk_*maxbin*",
    13: "clustCharge_ONTk_*maxbin*",
    14: "clustCharge_ONTk_*mpv*",
    15: "clustCharge_ONTk_*gausmean*",
}

ylow=[0,0,0,0,0,0,0,0,-10,0,10,10,0,0,0,0]
yhigh = [0,2500E3,6000,2500E3,2000,150,400,90,50,40,140,50,30,50,50,50] 

#easier to use to write out the file names (has to match plotlist without *
obsName = {
    1: "NClust_OFFTk_count",
    2: "NClust_OFFTk_mean",
    3: "NClust_ONTk_count",
    4: "NClust_ONTk_mean",
    5: "adc_ONTk_",
    6: "centralCharge_OFFTk_mpv",
    7: "centralStoN_OFFTk_mean",
    8: "centralStoN_OFFTk_mpv",
    9: "centralStoN_ONTk_mean",
    10: "centralStoN_ONTk_mpv",
    11: "charge_ONTk_",
    12: "clustCharge_OFFTk_maxbin",
    13: "clustCharge_ONTk_maxbin",
    14: "clustCharge_ONTk_mpv",
    15: "clustCharge_ONTk_gausmean",
}

##For time being, this is dir...
PD  = { 1 : "19Nov2011/Run2011B/MinimumBias"}

PDcompare  = False
obscompare = False

if obscompare :
    obs = { 1 : "centralStoN_ONTk_TIB_Layer1",
            2 : "centralStoN_ONTk_TIB_Layer2",
            3 : "centralStoN_ONTk_TIB_Layer3",
            4 : "centralStoN_ONTk_TIB_Layer4"
            }

if PDcompare:
    PD = { 1 : "SingleMu_Prompteco",
           4 : "SingleMu_ReReco"
           }
           

# Get file
#if 1 == 1 :
for k  in plotlist:
    print k
    for i in PD:
        j=1
        filelist =  os.popen("ls fig/"+PD[i]+"/"+plotlist[k]+".root").readlines()
        leg1 = TLegend(0.647651,0.7167832,0.9530201,0.9335664,"","NDC")
        mg_final = TMultiGraph()
        for file in filelist:
            #print file[0:-1]
            startD = int(string.find(file,plotlist[k][0:6]))
            endD = int(string.find(file,".root"))
            obs[j]= file[startD:endD]
            #print obs[j]
            f[i,j] = TFile(file[0:-1])
            # Get Canvas
            can[i,j] = f[i,j].Get("trendplot")
            # Get list of primitives
            lop[i,j] = can[i,j].GetListOfPrimitives()
            # Get multi graph object
            mgo[i,j] = lop[i,j].At(1)
            h[i,j]=mgo[i,j].GetHistogram()
            # Get list of graphs
            log[i,j]=mgo[i,j].GetListOfGraphs()
            # Get graph
            g[i,j] = log[i,j].At(1)
            g[i,j].SetHistogram(h[i,j])
            # Graph settings
            title = re.split("_",obs[j])[0]
            #print title
            #print obs[j]
            g[i,j].SetName(g[i,j].GetName()+"_"+PD[i]+"_"+obs[j])
            g[i,j].GetYaxis().SetTitleOffset(1.2);
            g[i,j].GetYaxis().SetLabelSize(0.03);
            g[i,j].GetYaxis().SetRangeUser(ylow[k],yhigh[k]);
            g[i,j].SetTitle(obsName[k]+": "+PD[1].replace("/","-"))
            g[i,j].SetMarkerStyle(24+i*j)
            g[i,j].SetMarkerSize(1.0)
            g[i,j].SetMarkerColor(i*j)
            # Add all graphs to final multigraph object
            mg_final.Add(g[i,j],"p")
            label = re.split("_",obs[j])[-3] + " " + re.split("_",obs[j])[-2] + " " + re.split("_",obs[j])[-1]
            leg1.AddEntry( g[i,j],label, "P")
            j=j+1
        C=TCanvas(obsName[k],obsName[k],600,600)
        C.SetFillColor(0)
        h[1,1].Draw()
        mg_final.Draw()
        leg1.SetFillColor(0)
        leg1.SetShadowColor(0)
        
        leg1.Draw()
    
       

#pt = C.GetPrimitive("title")
#pt.SetTextSize(5.0)
#pt.SetLineColor(0)
#pt.SetTextAlign(13)


#file=TFile("test.root","RECREATE")
#C.Write()


# histo doesnt work?
#h2=mgo2.GetHistogram()
        if   PDcompare  : C.SaveAs("PDcomparison_"+obsName[k]+".png")
        elif obscompare : C.SaveAs("obscomparison.root")
        else            : C.SaveAs("Compare_"+obsName[k]+".png")




##########

#PD = { #1 : "Jet_PromptReco",
#    #2 : "SingleElectron_PromptReco",
#    1 : "SingleMu_PromptReco",
#    #4 : "HLTMON_Express",
#    2 : "SingleMu_ReReco"
#    }
