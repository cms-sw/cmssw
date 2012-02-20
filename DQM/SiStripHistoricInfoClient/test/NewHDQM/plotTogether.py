from ROOT import *

f = {}
h = {}
can = {}
lop = {}
mgo = {}
log = {}
g   = {}

obs = {1 : "meanTrackChisquared"}
PD  = { 1 : "Jet"}

PDcompare  = True
obscompare = False

if obscompare :
    obs = { 1 : "centralStoN_TOB",
            2 : "centralStoN_TIB",
            3 : "centralStoN_TID1",
            4 : "centralStoN_TID2",
            5 : "centralStoN_TEC1",
            6 : "centralStoN_TEC2",
            7 : "meanTrackChisquared"
            }

if PDcompare:
    PD = { 1 : "SingleMu_PromptReco",
           4 : "SingleMu_ReReco"
           }
           
mg_final = TMultiGraph()

# Get file
for i in PD:
    for j in obs:
        f[i,j] = TFile("fig/"+PD[i]+"/"+obs[j]+".root")
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
        g[i,j].SetName(g[i,j].GetName()+"_"+PD[i]+"_"+obs[j])
        g[i,j].SetMarkerStyle(24+i*j)
        g[i,j].SetMarkerSize(2)
        g[i,j].SetMarkerColor(i*j+5)
        # Add all graphs to final multigraph object
        mg_final.Add(g[i,j],"lcp")
    
C=TCanvas("C","C",600,600)
h[1,1].Draw()
mg_final.Draw()

leg1 = TLegend(0.50, 0.50, 0.85, 0.95-0.20, "", "NDC")
for i in PD :
    for j in obs :
        leg1.AddEntry( g[i,j], PD[i]+"_"+obs[j], "PLE")

leg1.SetFillColor(0)
leg1.SetShadowColor(0)

leg1.Draw()


#file=TFile("test.root","RECREATE")
#C.Write()


# histo doesnt work?
#h2=mgo2.GetHistogram()

if   PDcompare  : C.SaveAs("PDcomparison_"+obs[1]+".png")
elif obscompare : C.SaveAs("obscomparison_"+PD[1]+".png") 
else            : C.SaveAs(PD[1]+"_"+obs[1]+".png")




##########

#PD = { #1 : "Jet_PromptReco",
#    #2 : "SingleElectron_PromptReco",
#    1 : "SingleMu_PromptReco",
#    #4 : "HLTMON_Express",
#    2 : "SingleMu_ReReco"
#    }
