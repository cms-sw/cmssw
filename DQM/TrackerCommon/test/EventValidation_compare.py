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

channel = "El"

from ROOT import *
import re

i=11

file = open('interestingEvents_'+channel+'.txt', 'r')
events = file.readlines()
runinfo_reference = { "Mu": '172268_Mu',
                      "El": '172268_El'}
runinfo_interest  = re.split('[:]',str(events[i]))

print "runinfo reference = ", runinfo_reference 
print "runinfo interest  = ", runinfo_interest 

runnumber_reference  = runinfo_reference[channel]
runnumber_interest   = runinfo_interest[0]
eventnumber_interest = runinfo_interest[2][:-1]

#################
### filenames ###
#################

#filename_interest  = "Playback_V0001_SiStrip_R000167102.root"
#filename_interest  = "/tmp/tdupree/Playback_V0001_Pixel_R000163796.root"
#filename_reference = "Playback_V0001_SiStrip_R000172268.root"
#filename_reference = "/tmp/tdupree/Playback_V0001_Pixel_R000172268.root"

#165970:168:184216723

filename_interest  = "DQM_V0001_R000"+str(runnumber_interest)+"__Global__CMSSW_X_Y_Z__RECO.root"
filename_reference = "DQM_V0001_R000"+str(runnumber_reference)+"__Global__CMSSW_X_Y_Z__RECO.root"

print " filename_interest  = ", filename_interest
print " filename_reference = ", filename_reference

filedir_interest  = "./FinalOutput/Event"+eventnumber_interest+"/"
filedir_reference = "./reference/"

#################
### get files ###
#################

file_interest  = filedir_interest +filename_interest
file_reference = filedir_reference+filename_reference

print "getting interesting sample from ... ", file_interest
tfile_interest  = TFile(file_interest) 
print "getting reference sample from ... ", file_reference
tfile_reference = TFile(file_reference)

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
    "Strip_nBadActiveChannel",
    "Strip_nFED"          ,
    "Strip_StoN_detFracMap",
    "Pixel_DigiOccupancy" ,
    "Pixel_nDigis_Barrel" ,
    "Pixel_ClusCharge_Endcap" ,
    "Pixel_ClusPosition_Disk" ,
    "Pixel_ClusCharge_OnTrack",
    "Pixel_ClusSize_OnTrack",
    "Pixel_ErrorType_vs_FedNr",
    "Track_NumberOfBarrelLayersPerTrack",
    "Track_NumberOfEndcapLayersPerTrack",
    "Track_NumberOfBarrelLayersPerTrackVsEtaProfile",
    "Track_NumberOfEndcapLayersPerTrackVsEtaProfile",
    "Track_GoodTrkNum"    ,
    "Track_GoodTrkEta"    ,
    "Track_GoodTrkPt"     ,
    "Track_nRecHitsPerTrack",
    "Track_GoodTrkChi2oNDF",
   
    ]

####################################
### the paths to the observables ###
####################################

interest_string  = "DQMData/Run " + run_interest
reference_string = "DQMData/Run " + run_reference

stringlist = {
    "Strip_nFED"              : "/SiStrip/Run summary/ReadoutView/FedMonitoringSummary/nFEDErrors"                                    ,
    "Strip_nBadActiveChannel" : "/SiStrip/Run summary/ReadoutView/FedMonitoringSummary/BadActiveChannelStatusBits"        ,
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
    "Track_nRecHitsPerTrack"  : "/Tracking/Run summary/TrackParameters/HitProperties/GoodTrackNumberOfRecHitsPerTrack_GenTk"          ,
    "Track_GoodTrkChi2oNDF"   : "/Tracking/Run summary/TrackParameters/GeneralProperties/GoodTrackChi2oNDF_GenTk"                     ,
    "Pixel_DigiOccupancy"     : "/Pixel/Run summary/averageDigiOccupancy"                                                             ,
    "Pixel_ErrorRate"         : "/Pixel/Run summary/AdditionalPixelErrors/errorRate"                                                  ,
    "Pixel_nDigis_Barrel"     : "/Pixel/Run summary/Barrel/SUMOFF_ndigis_Barrel"                                                      ,
    "Pixel_ErrorType_vs_FedNr": "/Pixel/Run summary/AdditionalPixelErrors/FedETypeNErrArray"                                          ,
    "Pixel_ClusCharge_OnTrack": "/Pixel/Run summary/Clusters/OnTrack/charge_siPixelClusters"                                          , 
    "Pixel_ClusSize_OnTrack"  : "/Pixel/Run summary/Clusters/OnTrack/size_siPixelClusters"                                            , 
    "Track_NumberOfBarrelLayersPerTrack" : "/Tracking/Run summary/TrackParameters/HitProperties/NumberOfPixBarrelLayersPerTrack_GenTk",
    "Track_NumberOfEndcapLayersPerTrack" : "/Tracking/Run summary/TrackParameters/HitProperties/NumberOfPixEndcapLayersPerTrack_GenTk",
    "Track_NumberOfBarrelLayersPerTrackVsEtaProfile" :"/Tracking/Run summary/TrackParameters/HitProperties/NumberOfPixBarrelLayersPerTrackVsEtaProfile_GenTk",
    "Track_NumberOfEndcapLayersPerTrackVsEtaProfile" :"/Tracking/Run summary/TrackParameters/HitProperties/NumberOfPixEndcapLayersPerTrackVsEtaProfile_GenTk",
    "Track_GoodTrkNum"        : "/Tracking/Run summary/TrackParameters/GeneralProperties/NumberOfGoodTracks_GenTk"                    ,
    "Track_GoodTrkEta"        : "/Tracking/Run summary/TrackParameters/GeneralProperties/GoodTrackEta_ImpactPoint_GenTk"              ,
    "Track_GoodTrkPt"         : "/Tracking/Run summary/TrackParameters/GeneralProperties/GoodTrackPt_ImpactPoint_GenTk"               ,
    "Pixel_ClusCharge_Endcap" : "/Pixel/Run summary/Endcap/SUMOFF_nclusters_Endcap"                                                   , 
    "Pixel_ClusPosition_Disk" : "/Pixel/Run summary/Clusters/OnTrack/position_siPixelClusters_mz_Disk_1"                               ,
#    "Pixel_ClusSize_Endcap"   : "/Pixel/Run summary/Clusters/SUMCLU_charge_Endcap"                                                  
    }

#######################################################
### plot all observables, together for both samples ###
#######################################################

gROOT.SetStyle("Plain")

th1_interest = {}
th1_reference = {}
C = {}

gStyle.SetPalette(1)


myf = TFile(runnumber_interest+"_"+eventnumber_interest+".root",
            "RECREATE")

for var in varNameList: 
    th1_interest[var]  = tfile_interest.Get(  interest_string + stringlist[var] )
    th1_reference[var] = tfile_reference.Get(reference_string + stringlist[var] )

    print "*** variable = " , var  
    print "*** path     = " , stringlist[var]  

    C[var]=TCanvas("C"+var,"C"+var,1100,800)
    C[var].Divide(3,2)

    C[var].cd(1)
    th1_reference[var].Draw()
    
    C[var].cd(2)
    print "th1_reference[var].GetEntries() = ", th1_reference[var].GetEntries()
    print "th1_interest[var].GetEntries()  = ", th1_interest[var].GetEntries()
    normfactor=1
    if th1_interest[var].GetEntries() : normfactor = th1_reference[var].GetEntries()/th1_interest[var].GetEntries()
    print "normfactor = ", normfactor 
    th1_interest[var].Scale(normfactor)
    th1_interest[var].Draw()

    C[var].cd(3)
    th1_reference[var].SetLineColor(3)
    th1_reference[var].Draw()
    th1_interest[var].SetLineColor(2)
    th1_interest[var].Draw("same")

    C[var].cd(4)
    th1_reference[var].Draw("colz")
    
    C[var].cd(5)
    th1_interest[var].Draw("colz")

    C[var].Write("C"+var)
    th1_interest[var].Write(  "th1_interest_"  + stringlist[var])
    th1_reference[var].Write( "th1_reference_" + stringlist[var])
    
myf.Close()
myf.Write()

print "===> now let's have a look at the distributions!!"

###############
### THE END ###
###############    
