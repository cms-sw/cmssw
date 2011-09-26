###########################################
###                                     ###
### Script to compare two DQM-files     ###
###                                     ###
### 1) sample with "interesting events" ###
### 2) reference sample                 ###
###                                     ###
### DQM-files are in standard format,   ###
### produced using sistrip_dqm*.py      ### 
###                                     ###
### To do: let user specify filenames,  ###
### parsing them using os               ###
###                                     ###
###########################################

from ROOT import *

#################
### filenames ###
#################

#filename_interest  = "Playback_V0001_SiStrip_R000167102.root"
#filename_interest  = "/tmp/tdupree/Playback_V0001_Pixel_R000163796.root"
#filename_reference = "Playback_V0001_SiStrip_R000172268.root"
#filename_reference = "/tmp/tdupree/Playback_V0001_Pixel_R000172268.root"

filename_interest  = "DQM_V0001_R000163796__Global__CMSSW_X_Y_Z__RECO.root"
filename_reference = "DQM_V0001_R000172268__Global__CMSSW_X_Y_Z__RECO.root"

print " filename_interest  = ", filename_interest
print " filename_reference = ", filename_reference

#################
### get files ###
#################

tfile_interest  = TFile(filename_interest)
tfile_reference = TFile(filename_reference)

run_interest  =  filename_interest[14:20]
run_reference = filename_reference[14:20]

print " run_interest  = ", run_interest
print " run_reference = ", run_reference

#######################################
### the list of monitored variables ###
#######################################

varNameList = [
    "Strip_StoN_off_TEC1" ,
    "Strip_StoN_off_TEC2" ,
    "Strip_StoN_off_TID1" ,
    "Strip_StoN_off_TID2" ,
    "Strip_StoN_off_TIB"  ,
    "Strip_StoN_off_TOB"  ,
    "Strip_StoN_on_TEC1"  ,
    "Strip_StoN_on_TEC2"  ,
    "Strip_StoN_on_TID1"  ,
    "Strip_StoN_on_TID2"  ,
    "Strip_StoN_on_TIB"   ,
    "Strip_StoN_on_TOB"   ,
    "Strip_StoN_detFracMap",
    "Strip_nFED"          ,
    "Track_GoodTrkNum"    ,
    "Track_GoodTrkEta"    ,
    "Track_GoodTrkPt"     ,
    "Pixel_DigiOccupancy" ,
    "Pixel_ErrorRate"     ,
    "Pixel_nDigis_Barrel" ,
    "Pixel_ClusCharge_Endcap" ,
    "Pixel_ClusSize_Endcap"
               ]

####################################
### the paths to the observables ###
####################################

interest_string  = "DQMData/Run " + run_interest
reference_string = "DQMData/Run " + run_reference

stringlist = {
    "Strip_nFED"              : "/SiStrip/Run summary/ReadoutView/FedMonitoringSummary/nFEDErrors"                                    ,
    "Strip_StoN_on_TEC1"      : "/SiStrip/Run summary/MechanicalView/TEC/side_1/Summary_ClusterStoNCorr_OnTrack__TEC__side__1"        ,
    "Strip_StoN_on_TEC2"      : "/SiStrip/Run summary/MechanicalView/TEC/side_2/Summary_ClusterStoNCorr_OnTrack__TEC__side__2"        ,
    "Strip_StoN_on_TID1"      : "/SiStrip/Run summary/MechanicalView/TID/side_1/Summary_ClusterStoNCorr_OnTrack__TID__side__1"        ,
    "Strip_StoN_on_TID2"      : "/SiStrip/Run summary/MechanicalView/TID/side_2/Summary_ClusterStoNCorr_OnTrack__TID__side__2"        ,
    "Strip_StoN_on_TIB"       : "/SiStrip/Run summary/MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack__TIB"                        ,
    "Strip_StoN_on_TOB"       : "/SiStrip/Run summary/MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack__TOB"                        ,
    "Strip_StoN_off_TEC1"     : "/SiStrip/Run summary/MechanicalView/TEC/side_1/Summary_TotalNumberOfClusters_OffTrack__TEC__side__1" ,
    "Strip_StoN_off_TEC2"     : "/SiStrip/Run summary/MechanicalView/TEC/side_2/Summary_TotalNumberOfClusters_OffTrack__TEC__side__2" ,
    "Strip_StoN_off_TID1"     : "/SiStrip/Run summary/MechanicalView/TID/side_1/Summary_TotalNumberOfClusters_OffTrack__TID__side__1" ,
    "Strip_StoN_off_TID2"     : "/SiStrip/Run summary/MechanicalView/TID/side_2/Summary_TotalNumberOfClusters_OffTrack__TID__side__2" ,
    "Strip_StoN_off_TIB"      : "/SiStrip/Run summary/MechanicalView/TIB/Summary_TotalNumberOfClusters_OffTrack__TIB"                 ,
    "Strip_StoN_off_TOB"      : "/SiStrip/Run summary/MechanicalView/TOB/Summary_TotalNumberOfClusters_OffTrack__TOB"                 ,
    "Strip_StoN_detFracMap"   : "/SiStrip/Run summary/MechanicalView/detFractionReportMap"                                            ,
    "Track_GoodTrkNum"        : "/Tracking/Run summary/TrackParameters/GeneralProperties/NumberOfGoodTracks_GenTk"                    ,
    "Track_GoodTrkEta"        : "/Tracking/Run summary/TrackParameters/GeneralProperties/GoodTrackEta_ImpactPoint_GenTk"              ,
    "Track_GoodTrkPt"         : "/Tracking/Run summary/TrackParameters/GeneralProperties/GoodTrackPt_ImpactPoint_GenTk"               ,
    "Pixel_DigiOccupancy"     : "/Pixel/Run summary/averageDigiOccupancy"                                                             ,
    "Pixel_ErrorRate"         : "/Pixel/Run summary/AdditionalPixelErrors/errorRate"                                                  ,
    "Pixel_nDigis_Barrel"     : "/Pixel/Run summary/Barrel/SUMOFF_ndigis_Barrel"                                                      ,                      
    "Pixel_ClusCharge_Endcap" : "/Pixel/Run summary/Endcap/SUMOFF_size_OnTrack_Endcap"                                                        , 
    "Pixel_ClusSize_Endcap"   : "/Pixel/Run summary/Clusters/SUMCLU_charge_Endcap"                                                  
    }

#######################################################
### plot all observables, together for both samples ###
#######################################################

gROOT.SetStyle("Plain")

th1_interest = {}
th1_reference = {}
C = {}

for var in varNameList: 
    th1_interest[var]  = tfile_interest.Get(  interest_string + stringlist[var] )
    th1_reference[var] = tfile_reference.Get(reference_string + stringlist[var] )

    C[var]=TCanvas("C"+var,"C"+var,1100,400)
    C[var].Divide(3)

    C[var].cd(1)
    th1_reference[var].Draw()

    C[var].cd(2)
    print "*** variable = " , var  
    print "th1_reference[var].GetEntries() = ", th1_reference[var].GetEntries()
    print "th1_interest[var].GetEntries()  = ", th1_interest[var].GetEntries()
    normfactor = th1_reference[var].GetEntries()/th1_interest[var].GetEntries()
    print "normfactor = ", normfactor 
    th1_interest[var].Scale(normfactor)
    th1_interest[var].Draw()

    C[var].cd(3)
    th1_reference[var].SetLineColor(3)
    th1_reference[var].Draw()
    th1_interest[var].SetLineColor(2)
    th1_interest[var].Draw("same")

print "===> now let's have a look at the distributions!!"

###############
### THE END ###
###############    
