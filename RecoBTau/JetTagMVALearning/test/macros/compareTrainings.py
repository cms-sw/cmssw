######################################################################
###                                                                ###
### Script for plotting comparisons of different performances      ###
###                                                                ###
### Main thing to be changed by YOU: list of paths.                ###
###                                                                ###
### Easily adjustable                                              ###
### - legend                                                       ###
### - color                                                        ### 
###                                                                ###
### Assumes you have your CSV performance files                    ###
### in different directories for different training settings       ###
###                                                                ###
### TODO: - make it usable for pT/eta comparisons too              ###
###       - let setting be specified 'outside' the script when running
###                                                                ###
######################################################################
###                                                                ###
### What the user needs to do:                                     ###
###                                                                ###
### 1) Adjust the list of training settings that you would like to compare
### 2) Update the maps of paths/colors/legends such that the maps are not empty for you new variable
### 3) Run as python -i compareTrainings_NEW.py                    ###
###                                                                ###
######################################################################

from ROOT import *
gROOT.SetStyle("Plain")

############################################################################################ 
### This is the list of training settings you would like to compare                      ###
### => Change it if you would like to compare more/different settings.                   ###
### => If you add new variables, update the maps of path/color/legend accordingly (once) ###
############################################################################################ 

settingList = ["old",
               "default",
               "deltar",
               "all"
               ]

#################################################
### here are just some maps of paths          ###
###  and legend names and colors for plotting ###
### - paths: adjust only once                 ### 
### - legend/color: adjust when you want      ###
#################################################

path = "./testBtagVal_release/CMSSW_4_4_4/src/UserCode/PetraVanMulders/BTagging/CSVLR_default/"


pathList = {"old"     : "/home/fynu/tdupree/scratch/testBtagVal/CMSSW_4_4_4/src/Validation/RecoB/test/",
            "deltar"  : path+"CSV_trackDeltaR",
            "default" : path+"CSV_default",
            "all"     : path+"CSV_all"
            } 

leg = {"old"     : "Old",
       "deltar"  : "Delta R",
       "default" : "Default",
       "all"     : "All"
       }

color = {"old"     : 4,
         "default" : 7,
         "deltar"  : 3,
         "all"     : 6
         }

#####################################
### Now we're gonna loop and plot ###
#####################################
   
plotList   = {}
plotList_C = {}

for setting in settingList:
    ### Get file and plot
    fileName = pathList[setting]+"/DQM_V0001_R000000001__POG__BTAG__categories.root"
    file     = TFile.Open(fileName)
    plot     = file.Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL")
    plot_C   = file.Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_C_discr_CSV_GLOBAL")
    ### Set name of plot
    plot.SetName(pathList[setting])
    plot_C.SetName(pathList[setting])
    ### Fill plot list
    plotList[setting]   = plot  
    plotList_C[setting] = plot_C


### Make canvas ###
        
Plots = TCanvas("Plots","",1200,600)
Plots.Divide(2)

Plots.cd(1).SetLogy()
leg1 = TLegend(0.2,0.6,0.45,0.9)
leg1.SetFillColor(0);

### and draw ###

first=true
for setting in settingList:
    plotList[setting].SetMarkerColor(color[setting])
    if first:
        plotList[setting].GetXaxis().SetTitle("B efficiency")
        plotList[setting].GetYaxis().SetTitle("DUSG efficiency")
        plotList[setting].SetTitle("")
        plotList[setting].Draw()
        first=false
    else         :
        plotList[setting].Draw("same")  
    leg1.AddEntry(plotList[setting],leg[setting],"p")
  
leg1.Draw()

###################
### Second plot ###
###################

Plots.cd(2).SetLogy()

leg2 = TLegend(0.2,0.6,0.45,0.9)
leg2.SetFillColor(0);

### and draw again

first=true
for setting in settingList:
    plotList_C[setting].SetMarkerColor(color[setting])
    if first:
        plotList_C[setting].Draw()
        plotList_C[setting].GetXaxis().SetTitle("B efficiency")
        plotList_C[setting].GetYaxis().SetTitle("C efficiency")
        plotList_C[setting].SetTitle("")
        first=false
    else         :
        plotList_C[setting].Draw("same")  
    leg2.AddEntry(plotList_C[setting],leg[setting],"p")
  
leg2.Draw()

###########
### FIN ###
###########
