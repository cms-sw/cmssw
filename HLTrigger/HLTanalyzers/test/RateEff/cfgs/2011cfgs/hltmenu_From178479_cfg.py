#------------------------------------------------------
# Configuration file for Rate & Efficiency calculations
#------------------------------------------------------
# This version is compliant with RateEff-02-XX-XX
# using logical parser for L1 seeds
#

##########################################
# General Menu & Run conditions
##########################################
run:{
    nEntries = -1;
    nPrintStatusEvery = 10000; # print out status every n events processed
    menuTag  = "HLT_Menu";
    alcaCondition = "startup";
    versionTag  = "20111026_DS_All"; 
    isRealData = true;
    doPrintAll = true;
    doDeterministicPrescale =true;
    dsList = "Datasets.list";
    readRefPrescalesFromNtuple = true;

};

########################################## 
# Run information for real data 
########################################## 
data:{ 
 # Enter the length of 1 lumi section and prescale factor of the dataset
 lumiSectionLength = 23.3;
 lumiScaleFactor = 1;
 prescaleNormalization = 1;

##run 178479, column 2
runLumiblockList = ( 
   (178479, 190, 636 ) # (runnr, minLumiBlock, maxLumiBlock)
  );

##column 1 
## runLumiblockList = ( 
##    (178479, 70, 185 ) # (runnr, minLumiBlock, maxLumiBlock)
##   );


};


##########################################
# Beam conditions
##########################################
beam:{
 bunchCrossingTime = 50.0E-09; # Design: 25 ns Startup: 75 ns
 iLumi = 3E33;
 maxFilledBunches = 3564;
 nFilledBunches = 800;
 cmsEnergy = 7.; # Collision energy in TeV
};

##########################################
# Samples & Processes
##########################################
process:{
 isPhysicsSample = [0]; #Must be an int type
 names = ["minbias"];
 fnames = ["openhlt_*.root"];

## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__BTag_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__Commissioning_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__Cosmics_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__DoubleElectron_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__DoubleMu_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__ElectronHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__HT_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__Jet_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MinimumBias_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MET_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MuOnia_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MultiJet_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MuEG_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__MuHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__Photon_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__PhotonHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__SingleElectron_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__SingleMu_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__Tau_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r178479__TauPlusX_Run2011B-PromptReco-v1__20111019_1605/"];



## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__BTag_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__Commissioning_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__Cosmics_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__DoubleElectron_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__DoubleMu_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__ElectronHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__HT_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__Jet_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MinimumBias_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MET_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MuOnia_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MultiJet_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MuEG_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__MuHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__Photon_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__PhotonHad_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__SingleElectron_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__SingleMu_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__Tau_Run2011B-PromptReco-v1__20111019_1605/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r178479__TauPlusX_Run2011B-PromptReco-v1__20111019_1605/"];


 doMuonCuts = [false];
 doElecCuts = [false];
 sigmas = [9.87E08]; # xsecs * filter efficiencies for QCD 15
};


##########################################
# Menu
##########################################
menu:{
 isL1Menu = false; # Default is false: is HLT Menu
 doL1preloop = true; 

## preFilterByBits = "HLT_Mu15_v8";


  # (TriggerName, Prescale, EventSize)
 triggers = (
#
## ############# dataset BTag ###############
##    ("HLT_BTagMu_DiJet20_Mu5_v13", "L1_Mu3_Jet16_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet40_Mu5_v13", "L1_Mu3_Jet20_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet70_Mu5_v13", "L1_Mu3_Jet28_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet110_Mu5_v13", "L1_Mu3_Jet28_Central", 1, 0.15)#,
## ############# dataset Commissioning ###############
##    ("HLT_Activity_Ecal_SC7_v8", "L1_ZeroBias_Ext", 1, 0.15),
##    ("HLT_L1SingleJet16_v4", "L1_SingleJet16", 1, 0.15),
##    ("HLT_L1SingleJet36_v4", "L1_SingleJet36", 1, 0.15),
##    ("HLT_L1SingleMuOpen_v4", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_L1SingleMuOpen_DT_v4", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_L1SingleMu10_v4", "L1_SingleMu10", 1, 0.15),
##    ("HLT_L1SingleMu20_v4", "L1_SingleMu20", 1, 0.15),
##    ("HLT_L2Mu10_v6", "L1_SingleMu10", 1, 0.15),
##    ("HLT_L2Mu20_v6", "L1_SingleMu12", 1, 0.15),
##    ("HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v12", "L1_SingleMu5_Eta1p5_Q80", 1, 0.15),
##    ("HLT_L1SingleEG5_v3", "L1_SingleEG5", 1, 0.15),
##    ("HLT_L1SingleEG12_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_BeamGas_HF_v6", "L1_BeamGas_Hf", 1, 0.15),
##    ("HLT_BeamGas_HF_Beam1_v2", "L1_BeamGas_Hf_BptxPlusPostQuiet", 1, 0.15),
##    ("HLT_BeamGas_HF_Beam2_v2", "L1_BeamGas_Hf_BptxMinusPostQuiet", 1, 0.15),
##    ("HLT_L1_PreCollisions_v3", "L1_PreCollisions", 1, 0.15),
##    ("HLT_L1_Interbunch_BSC_v3", "L1_InterBunch_Bsc", 1, 0.15),
##    ("HLT_IsoTrackHE_v9", "L1_SingleJet68", 1, 0.15),
##    ("HLT_IsoTrackHB_v8", "L1_SingleJet68", 1, 0.15)#,
## ############# dataset Cosmics ###############
##    ("HLT_BeamHalo_v7", "L1_BeamHalo", 1, 0.15),
##    ("HLT_L1SingleMuOpen_AntiBPTX_v3", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_L1TrackerCosmics_v4", "L1Tech_RPC_TTU_pointing_Cosmics.v0", 1, 0.15),
##    ("HLT_RegionalCosmicTracking_v8", "L1Tech_RPC_TTU_pointing_Cosmics.v0 AND L1_SingleMuOpen", 1, 0.15)#,
## ############# dataset DoubleElectron ###############
##    ("HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v10", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Ele8_v9", "L1_SingleEG5", 1, 0.15),
##    ("HLT_Ele8_CaloIdL_CaloIsoVL_v9", "L1_SingleEG5", 1, 0.15),
##    ("HLT_Ele8_CaloIdL_TrkIdVL_v9", "L1_SingleEG5", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v7", "L1_SingleEG5", 1, 0.15),
##    ("HLT_Ele17_CaloIdL_CaloIsoVL_v9", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v9", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v9", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v8", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Ele22_CaloIdL_CaloIsoVL_Ele15_HFT_v2", "L1_EG18_ForJet16", 1, 0.15),
##    ("HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v7", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_Ele17_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v11", "L1_SingleEG5", 1, 0.15),
##    ("HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v4", "L1_TripleEG7", 1, 0.15),
##    ("HLT_TripleEle10_CaloIdL_TrkIdVL_v10", "L1_TripleEG7", 1, 0.15)#,
## ############# dataset DoubleMu ###############
##    ("HLT_L1DoubleMu0_v4", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_L2DoubleMu0_v7", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_L2DoubleMu23_NoVertex_v8", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_L2DoubleMu30_NoVertex_v4", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_L2DoubleMu45_NoVertex_v1", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_DoubleMu3_v13", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_DoubleMu5_v4", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_DoubleMu7_v11", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu45_v9", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_DoubleMu7_Acoplanarity03_v3", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_Mu13_Mu8_v10", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_Mu17_Mu8_v10", "L1_DoubleMu3p5", 1, 0.15),
##    ("HLT_Mu17_TkMu8_v3", "L1_DoubleMu_10_Open", 1, 0.15),
##    ("HLT_TripleMu5_v12", "L1_TripleMu0", 1, 0.15),
##    ("HLT_DoubleMu5_IsoMu5_v11", "L1_TripleMu0", 1, 0.15),
##    ("HLT_Mu8_Jet40_v14", "L1_Mu3_Jet20_Central", 1, 0.15)#,
## ############# dataset ElectronHad ###############
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_v4", "L1_DoubleEG5", 1, 0.15),
##    ("HLT_HT350_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT45_v10", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT50_v4", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_Ele60_CaloIdT_TrkIdT_v4", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_HT450_Ele60_CaloIdT_TrkIdT_v3", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v8", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v8", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v8", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT40_v5", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT50_v4", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R014_MR200_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R025_MR200_v4", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R029_MR200_v4", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Ele12_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R033_MR200_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v12", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_CentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_DiCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_TriCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralPFJet30_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_BTagIP_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_WP80_DiCentralPFJet25_PFMHT15_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_WP80_DiCentralPFJet25_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_WP80_DiPFJet25_Deta3_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_CaloIdVT_TrkIdT_DiCentralPFJet25_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_CaloIdVT_TrkIdT_DiPFJet25_Deta3_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_WP80_DiCentralPFJet25_PFMHT25_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_WP80_DiPFJet25_Deta3p5_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT150_v3", "L1_DoubleEG5_HTT75", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass8_HT200_v3", "L1_DoubleEG5_HTT75", 1, 0.15)#,
## ############# dataset HT ###############
   ("HLT_FatJetMass850_DR1p1_Deta2p0_v5", "L1_HTT100", 1, 0.15),
   ("HLT_DiJet130_PT130_v9", "L1_SingleJet68", 1, 0.15),
   ("HLT_DiJet160_PT160_v9", "L1_SingleJet92", 1, 0.15),
   ("HLT_HT150_v11", "L1_HTT50", 1, 0.15),
   ("HLT_HT200_v11", "L1_HTT75", 1, 0.15),
   ("HLT_HT250_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT250_AlphaT0p58_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT250_AlphaT0p60_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT250_AlphaT0p65_v2", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_v12", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_PFMHT55_v12", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_PFMHT65_v5", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_CentralJet30_BTagIP_v10", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_CentralJet30_BTagIP_PFMHT55_v12", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_CentralJet30_BTagIP_PFMHT65_v5", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_AlphaT0p54_v5", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_AlphaT0p55_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT300_AlphaT0p60_v2", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_MHT100_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_MHT110_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_L1FastJet_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_L1FastJet_MHT100_v1", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_L1FastJet_MHT110_v1", "L1_HTT100", 1, 0.15),
   ("HLT_HT350_AlphaT0p53_v10", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_MHT90_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_MHT100_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_L1FastJet_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_L1FastJet_MHT90_v1", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_L1FastJet_MHT100_v1", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_AlphaT0p51_v10", "L1_HTT100", 1, 0.15),
   ("HLT_HT400_AlphaT0p52_v5", "L1_HTT100", 1, 0.15),
   ("HLT_HT450_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT450_AlphaT0p51_v5", "L1_HTT100", 1, 0.15),
   ("HLT_HT500_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT550_v11", "L1_HTT100", 1, 0.15),
   ("HLT_HT600_v4", "L1_HTT100", 1, 0.15),
   ("HLT_HT650_v4", "L1_HTT100", 1, 0.15),
   ("HLT_HT700_v2", "L1_HTT100", 1, 0.15),
   ("HLT_HT750_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT750_L1FastJet_v3", "L1_HTT100", 1, 0.15),
   ("HLT_HT2000_v5", "L1_HTT100", 1, 0.15),
   ("HLT_R014_MR150_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R020_MR150_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R020_MR550_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R025_MR150_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R025_MR450_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R033_MR350_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R038_MR250_v10", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_R038_MR300_v2", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
   ("HLT_RMR65_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15)#,
## ############# dataset Jet ###############
##    ("HLT_Jet30_v9", "L1_SingleJet16", 1, 0.15),
##    ("HLT_Jet30_L1FastJet_v3", "L1_SingleJet16", 1, 0.15),
##    ("HLT_Jet60_v9", "L1_SingleJet36", 1, 0.15),
##    ("HLT_Jet60_L1FastJet_v3", "L1_SingleJet36", 1, 0.15),
##    ("HLT_Jet110_v9", "L1_SingleJet68", 1, 0.15),
##    ("HLT_Jet190_v9", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet240_v9", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet240_L1FastJet_v3", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet300_v9", "L1_SingleJet128", 1, 0.15),
##    ("HLT_Jet300_L1FastJet_v3", "L1_SingleJet128", 1, 0.15),
##    ("HLT_Jet370_v10", "L1_SingleJet128", 1, 0.15),
##    ("HLT_Jet370_L1FastJet_v3", "L1_SingleJet128", 1, 0.15),
##    ("HLT_Jet370_NoJetID_v10", "L1_SingleJet128", 1, 0.15),
##    ("HLT_Jet800_v5", "L1_SingleJet128", 1, 0.15),
##    ("HLT_DiJetAve30_v9", "L1_SingleJet16", 1, 0.15),
##    ("HLT_DiJetAve60_v9", "L1_SingleJet36", 1, 0.15),
##    ("HLT_DiJetAve110_v9", "L1_SingleJet68", 1, 0.15),
##    ("HLT_DiJetAve190_v9", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve240_v9", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve300_v10", "L1_SingleJet128", 1, 0.15),
##    ("HLT_DiJetAve370_v10", "L1_SingleJet128", 1, 0.15)#,
## ############# dataset MinimumBias ###############
##    ("HLT_JetE30_NoBPTX_v8", "L1_SingleJet20_Central_NotBptxOR", 1, 0.15),
##    ("HLT_JetE30_NoBPTX_NoHalo_v10", "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_JetE30_NoBPTX3BX_NoHalo_v10", "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_JetE50_NoBPTX3BX_NoHalo_v5", "L1_SingleJet32_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_PixelTracks_Multiplicity80_v8", "L1_ETT220", 1, 0.15),
##    ("HLT_PixelTracks_Multiplicity100_v8", "L1_ETT220", 1, 0.15),
##    ("HLT_ZeroBias_v4", "L1_ZeroBias_Ext", 1, 0.15),
##    ("HLT_Physics_v2", "", 1, 0.15),
##    ("HLT_Random_v1", "", 1, 0.15)#,
## ############# dataset MET ###############
##    ("HLT_CentralJet80_MET65_v10", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET80_v9", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET95_v3", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET110_v3", "L1_ETM30", 1, 0.15),
##    ("HLT_DiJet60_MET45_v10", "L1_ETM20", 1, 0.15),
##    ("HLT_DiCentralJet20_MET100_HBHENoiseFiltered_v4", "L1_ETM30", 1, 0.15),
##    ("HLT_DiCentralJet20_MET80_v8", "L1_ETM30", 1, 0.15),
##    ("HLT_DiCentralJet20_BTagIP_MET65_v10", "L1_ETM30", 1, 0.15),
##    ("HLT_PFMHT150_v16", "L1_ETM30", 1, 0.15),
##    ("HLT_MET120_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_MET120_HBHENoiseFiltered_v6", "L1_ETM30", 1, 0.15),
##    ("HLT_MET200_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_MET200_HBHENoiseFiltered_v6", "L1_ETM30", 1, 0.15),
##    ("HLT_MET400_v2", "L1_ETM30", 1, 0.15),
##    ("HLT_R014_MR200_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R014_MR400_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R014_MR450_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R020_MR300_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R020_MR350_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R030_MR200_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_R030_MR250_CentralJet40_BTagIP_v3", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_L2Mu60_1Hit_MET40_v6", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_L2Mu60_1Hit_MET60_v6", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_Mu15_L1ETM20_v3", "L1_Mu10_ETM20", 1, 0.15),
##    ("HLT_IsoMu15_L1ETM20_v3", "L1_Mu10_ETM20", 1, 0.15)#,
## ############# dataset MuOnia ###############
##    ("HLT_DoubleMu4_Jpsi_Displaced_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_DoubleMu5_Jpsi_Displaced_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_DoubleMu4_Dimuon4_Bs_Barrel_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_DoubleMu4_Dimuon6_Bs_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_DoubleMu4p5_LowMass_Displaced_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_DoubleMu5_LowMass_Displaced_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon0_Omega_Phi_v3", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_v9", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_NoVertexing_v6", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon0_Upsilon_v9", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon6_LowMass_v4", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_Dimuon7_Upsilon_Barrel_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon9_Upsilon_Barrel_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon9_PsiPrime_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon10_Jpsi_Barrel_v9", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon11_PsiPrime_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon13_Jpsi_Barrel_v4", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_Muon_v10", "L1_TripleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Upsilon_Muon_v10", "L1_TripleMu0", 1, 0.15),
##    ("HLT_Mu5_L2Mu2_Jpsi_v12", "L1_DoubleMu0_HighQ", 1, 0.15),
##    ("HLT_Mu5_Track2_Jpsi_v12", "L1_SingleMu3", 1, 0.15),
##    ("HLT_Mu7_Track7_Jpsi_v13", "L1_SingleMu7", 1, 0.15)#,
## ############# dataset MultiJet ###############
##    ("HLT_DoubleJet30_ForwardBackward_v10", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet60_ForwardBackward_v10", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet70_ForwardBackward_v10", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet80_ForwardBackward_v10", "L1_DoubleForJet44_EtaOpp", 1, 0.15),
##    ("HLT_DiCentralJet36_BTagIP3DLoose_v4", "L1_DoubleJet36_Central", 1, 0.15),
##    ("HLT_CentralJet46_CentralJet38_DiBTagIP3D_v6", "L1_DoubleJet36_Central", 1, 0.15),
##    ("HLT_CentralJet60_CentralJet53_DiBTagIP3D_v5", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_QuadJet40_v11", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet45_DiJet40_v3", "L1_HTT100", 1, 0.15),
##    ("HLT_QuadJet50_DiJet40_v5", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet50_DiJet40_L1FastJet_v2", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet40_IsoPFTau40_v17", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet45_IsoPFTau45_v12", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet50_IsoPFTau50_v6", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet70_v10", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet80_v5", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet80_L1FastJet_v2", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_QuadJet90_v3", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_SixJet45_v3", "L1_HTT100", 1, 0.15),
##    ("HLT_SixJet45_L1FastJet_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_EightJet35_v3", "L1_HTT100", 1, 0.15),
##    ("HLT_EightJet35_L1FastJet_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_EightJet40_v3", "L1_HTT100", 1, 0.15),
##    ("HLT_EightJet40_L1FastJet_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_EightJet120_v5", "L1_QuadJet28_Central", 1, 0.15),
##    ("HLT_ExclDiJet60_HFOR_v9", "L1_SingleJet36", 1, 0.15),
##    ("HLT_ExclDiJet60_HFAND_v9", "L1_SingleJet36_FwdVeto", 1, 0.15),
##    ("HLT_L1DoubleJet36Central_v4", "L1_DoubleJet36_Central", 1, 0.15)#,
## ############# dataset MuEG ###############
##    ("HLT_Mu5_DoubleEle8_CaloIdT_TrkIdVL_v7", "L1_MuOpen_DoubleEG5", 1, 0.15),
##    ("HLT_Mu5_Ele8_CaloIdT_CaloIsoVL_v4", "L1_Mu3_EG5", 1, 0.15),
##    ("HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v7", "L1_MuOpen_DoubleEG5", 1, 0.15),
##    ("HLT_Mu8_Ele17_CaloIdL_v12", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v7", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu8_Photon20_CaloIdVT_IsoT_v12", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu15_Photon20_CaloIdL_v13", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu15_DoublePhoton15_CaloIdL_v13", "L1_MuOpen_DoubleEG5", 1, 0.15),
##    ("HLT_Mu17_Ele8_CaloIdL_v12", "L1_Mu7_EG5", 1, 0.15),
##    ("HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_v7", "L1_Mu12_EG5", 1, 0.15),
##    ("HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v7", "L1_DoubleMuOpen_EG5", 1, 0.15),
##    ("HLT_DoubleMu5_Ele8_CaloIdT_TrkIdT_v3", "L1_DoubleMuOpen_EG5", 1, 0.15)#,
## ############# dataset MuHad ###############
##    ("HLT_Mu10_R014_MR200_v4", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Mu10_R025_MR200_v5", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Mu10_R029_MR200_v5", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_Mu10_R033_MR200_v4", "L1_ETM30 OR L1_HTT50_HTM30", 1, 0.15),
##    ("HLT_HT300_Mu15_PFMHT40_v5", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_Mu15_PFMHT50_v4", "L1_HTT100 AND L1_SingleMuOpen", 1, 0.15),
##    ("HLT_HT350_Mu5_PFMHT45_v12", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_Mu5_PFMHT50_v4", "L1_HTT100 AND L1_SingleMuOpen", 1, 0.15),
##    ("HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT150_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu8_Ele8_CaloIdT_TrkIdVL_Mass8_HT200_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT150_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_TkIso10Mu5_Ele8_CaloIdT_CaloIsoVVL_TrkIdVL_Mass8_HT200_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu17_eta2p1_CentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_DiCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_TriCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_QuadCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_CentralJet30_BTagIP_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu12_eta2p1_DiCentralJet20_BTagIP3D1stTrack_v5", "L1_Mu10_Eta2p1_DoubleJet_16_8_Central", 1, 0.15),
##    ("HLT_Mu12_eta2p1_DiCentralJet20_DiBTagIP3D1stTrack_v5", "L1_Mu10_Eta2p1_DoubleJet_16_8_Central", 1, 0.15),
##    ("HLT_Mu40_HT300_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu60_HT300_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_CentralJet30_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_CentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_TriCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_QuadCentralPFJet30_v2", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_CentralJet30_BTagIP_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_DiCentralPFJet25_PFMHT15_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiCentralPFJet25_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT15_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiCentralPFJet25_PFMHT25_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Mu17_eta2p1_DiPFJet25_Deta3_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu17_eta2p1_DiPFJet25_Deta3_PFJet25_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_DoubleMu5_Mass8_HT150_v4", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_DoubleMu8_Mass8_HT150_v4", "L1_Mu0_HTT50 AND L1_SingleMuOpen", 1, 0.15),
##    ("HLT_DoubleMu8_Mass8_HT200_v4", "L1_Mu0_HTT50 AND L1_SingleMuOpen", 1, 0.15),
##    ("HLT_DoubleTkIso10Mu5_Mass8_HT150_v4", "L1_Mu0_HTT50 AND L1_SingleMuOpen", 1, 0.15),
##    ("HLT_DoubleTkIso10Mu5_Mass8_HT200_v4", "L1_Mu0_HTT50 AND L1_SingleMuOpen", 1, 0.15)#,
## ############# dataset Photon ###############
##    ("HLT_Photon20_CaloIdVL_IsoL_v8", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Photon20_R9Id_Photon18_R9Id_v7", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Photon26_Photon18_v7", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon26_CaloIdXL_IsoXL_Photon18_v3", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon26_CaloIdXL_IsoXL_Photon18_R9IdT_Mass60_v3", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon26_CaloIdXL_IsoXL_Photon18_CaloIdXL_IsoXL_Mass60_v3", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon26_R9IdT_Photon18_CaloIdXL_IsoXL_Mass60_v3", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon26_R9IdT_Photon18_R9IdT_Mass60_v1", "L1_DoubleEG_12_5", 1, 0.15),
##    ("HLT_Photon30_CaloIdVL_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon30_CaloIdVL_IsoL_v10", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_Photon22_v1", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_CaloIdVL_Photon22_CaloIdVL_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_CaloIdL_IsoVL_Photon22_v7", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_IsoVL_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_CaloIdL_IsoVL_Photon22_R9Id_v5", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_R9Id_Photon22_CaloIdL_IsoVL_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon36_R9Id_Photon22_R9Id_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon50_CaloIdVL_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon50_CaloIdVL_IsoL_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon75_CaloIdVL_v7", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon75_CaloIdVL_IsoL_v9", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon90_CaloIdVL_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon90_CaloIdVL_IsoL_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon135_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon225_NoHE_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon400_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon200_NoHE_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton43_HEVT_v1", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton48_HEVT_v1", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton70_v1", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton80_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton5_IsoVL_CEP_v8", "L1_DoubleEG2_FwdVeto", 1, 0.15),
##    ("HLT_DoubleEle33_CaloIdL_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoubleEle33_CaloIdL_CaloIsoT_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoubleEle33_CaloIdT_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoubleEle45_CaloIdL_v5", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton40_CaloIdL_MR150_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_DoublePhoton40_CaloIdL_R014_MR150_v3", "L1_SingleEG20", 1, 0.15)#,
## ############# dataset PhotonHad ###############
##    ("HLT_Photon60_CaloIdL_HT300_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon60_CaloIdL_MHT70_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdXL_HT400_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdXL_HT500_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdXL_MHT90_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdXL_MHT100_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet25_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon90EBOnly_CaloIdVL_IsoL_TriPFJet30_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R014_MR150_v1", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R017_MR500_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R023_MR350_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R029_MR250_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R042_MR200_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon55_CaloIdL_R017_MR500_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon55_CaloIdL_R023_MR350_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon55_CaloIdL_R029_MR250_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon55_CaloIdL_R042_MR200_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon30_CaloIdVT_CentralJet20_BTagIP_v6", "L1_SingleEG20", 1, 0.15)#,
## ############# dataset SingleElectron ###############
##    ("HLT_Ele25_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v6", "L1_SingleEG18", 1, 0.15),
##    ("HLT_Ele27_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_WP80_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele27_WP80_PFMT50_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_WP70_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_WP70_PFMT50_v8", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele65_CaloIdVT_TrkIdT_v5", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele80_CaloIdVT_TrkIdT_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele100_CaloIdVT_TrkIdT_v2", "L1_SingleEG20", 1, 0.15)#,
## ############# dataset SingleMu ###############
##    ("HLT_Mu5_v13", "L1_SingleMu3", 1, 0.15),
##    ("HLT_Mu8_v11", "L1_SingleMu3", 1, 0.15),
##    ("HLT_Mu12_v11", "L1_SingleMu7", 1, 0.15),
##    ("HLT_Mu15_v12", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu20_v11", "L1_SingleMu12", 1, 0.15),
##    ("HLT_Mu24_v11", "L1_SingleMu16", 1, 0.15),
##    ("HLT_Mu30_v11", "L1_SingleMu12", 1, 0.15),
##    ("HLT_Mu40_v9", "L1_SingleMu16", 1, 0.15),
##    ("HLT_Mu40_eta2p1_v4", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_Mu50_eta2p1_v1", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_Mu60_eta2p1_v4", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_Mu100_eta2p1_v4", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_Mu200_eta2p1_v1", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu15_v17", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu15_eta2p1_v4", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu20_v12", "L1_SingleMu12", 1, 0.15),
##    ("HLT_IsoMu24_v12", "L1_SingleMu16", 1, 0.15),
##    ("HLT_IsoMu24_eta2p1_v6", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu30_eta2p1_v6", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu34_eta2p1_v4", "L1_SingleMu16_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu40_eta2p1_v1", "L1_SingleMu16_Eta2p1", 1, 0.15)#,
## ############# dataset Tau ###############
##    ("HLT_MediumIsoPFTau35_Trk20_v5", "L1_SingleJet52_Central", 1, 0.15),
##    ("HLT_MediumIsoPFTau35_Trk20_MET60_v5", "L1_Jet52_Central_ETM30", 1, 0.15),
##    ("HLT_MediumIsoPFTau35_Trk20_MET70_v5", "L1_Jet52_Central_ETM30", 1, 0.15),
##    ("HLT_DoubleIsoPFTau45_Trk5_eta2p1_v7", "L1_DoubleTauJet44_Eta2p17 OR L1_DoubleJet64_Central", 1, 0.15),
##    ("HLT_DoubleIsoPFTau55_Trk5_eta2p1_v4", "L1_DoubleTauJet44_Eta2p17 OR L1_DoubleJet64_Central", 1, 0.15)#,
## ############# dataset TauPlusX ###############
##    ("HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v12", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_DoubleIsoPFTau10_Trk3_PFMHT50_v4", "L1_HTT100", 1, 0.15),
##    ("HLT_Mu15_LooseIsoPFTau15_v13", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu15_eta2p1_MediumIsoPFTau20_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_IsoMu15_eta2p1_TightIsoPFTau20_v5", "L1_SingleMu14_Eta2p1", 1, 0.15),
##    ("HLT_Ele18_CaloIdVT_TrkIdT_MediumIsoPFTau20_v5", "L1_SingleEG15", 1, 0.15),
##    ("HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v5", "L1_SingleEG18 OR L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau25_v4", "L1_SingleEG22", 1, 0.15)#,
## # 
 );

 # For L1 prescale preloop to be used in HLT mode only
 L1triggers = ( 
#
  ("L1_SingleJet52_Central", 1),
  ("L1_SingleJet36_FwdVeto", 1),
  ("L1_MuOpen_EG12", 1),
  ("L1_ETT220", 1),
  ("L1_Mu12_EG5", 1),
  ("L1_SingleMu16", 1),
  ("L1_SingleEG5", 1),
  ("L1_DoubleMu_10_Open", 1),
  ("L1_SingleJet16", 1),
  ("L1_SingleEG22", 1),
  ("L1_DoubleMu0_HighQ", 1),
  ("L1_SingleEG20", 1),
  ("L1_Mu3_Jet20_Central", 1),
  ("L1_DoubleMu3p5", 1),
  ("L1_DoubleForJet32_EtaOpp", 1),
  ("L1_SingleMuOpen", 1),
  ("L1_TripleEG7", 1),
  ("L1_HTT75", 1),
  ("L1Tech_HCAL_HF_MM_or_PP_or_PM.v0", 1),
  ("L1_ETM30", 1),
  ("L1_SingleMu14_Eta2p1", 1),
  ("L1_BeamHalo", 1),
  ("L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1),
  ("L1_Jet52_Central_ETM30", 1),
  ("L1_Mu10_Eta2p1_DoubleJet_16_8_Central", 1),
  ("L1_HTT50_HTM30", 1),
  ("L1_DoubleEG_12_5", 1),
  ("L1_SingleMu12", 1),
  ("L1_EG18_ForJet16", 1),
  ("L1Tech_RPC_TTU_pointing_Cosmics.v0", 1),
  ("L1_SingleMu5_Eta1p5_Q80", 1),
  ("L1_EG5_DoubleJet20_Central", 1),
  ("L1_BeamGas_Hf_BptxMinusPostQuiet", 1),
  ("L1_SingleMu20", 1),
  ("L1_DoubleEG5_HTT75", 1),
  ("L1_ETM20", 1),
  ("L1_SingleJet36", 1),
  ("L1_SingleJet52", 1),
  ("L1_SingleJet68", 1),
  ("L1_SingleJet92", 1),
  ("L1_SingleJet128", 1),
  ("L1_SingleMu3", 1),
  ("L1_SingleMu7", 1),
  ("L1_SingleMu10", 1),
  ("L1_SingleIsoEG12", 1),
  ("L1_SingleEG12", 1),
  ("L1_SingleEG15", 1),
  ("L1_SingleEG30", 1),
  ("L1_ZeroBias_Ext", 1),
  ("L1Tech_HCAL_HO_totalOR.v0", 1),
  ("L1Tech_HCAL_HBHE_totalOR.v0", 1),
  ("L1_SingleEG18", 1),
  ("L1_InterBunch_Bsc", 1),
  ("L1_Mu3_EG5", 1),
  ("L1_DoubleJet36_Central", 1),
  ("L1_Mu3_Jet16_Central", 1),
  ("L1_Mu3_Jet28_Central", 1),
  ("L1_BeamGas_Hf_BptxPlusPostQuiet", 1),
  ("L1_PreCollisions", 1),
  ("L1_EG5_HTT100", 1),
  ("L1_DoubleMu0", 1),
  ("L1_DoubleMu3", 1),
  ("L1_Mu10_ETM20", 1),
  ("L1_HTT100", 1),
  ("L1_SingleJet32_NotBptxOR_NotMuBeamHalo", 1),
  ("L1_Mu7_EG5", 1),
  ("L1_Mu0_HTT50", 1),
  ("L1_DoubleMuOpen_EG5", 1),
  ("L1_SingleMu16_Eta2p1", 1),
  ("L1_SingleJet20_Central_NotBptxOR", 1),
  ("L1_DoubleForJet44_EtaOpp", 1),
  ("L1_DoubleEG2_FwdVeto", 1),
  ("L1_TripleMu0", 1),
  ("L1_DoubleEG5", 1),
  ("L1_QuadJet28_Central", 1),
  ("L1_DoubleEG10", 1),
  ("L1_DoubleEG3", 1),
  ("L1_DoubleIsoEG10", 1),
  ("L1_SingleIsoEG12_Eta2p17", 1),
  ("L1_SingleMu25", 1),
  ("L1_DoubleMu5", 1),
  ("L1_BeamGas_Hf", 1),
  ("L1_DoubleTauJet44_Eta2p17", 1),
  ("L1_DoubleJet64_Central", 1),
  ("L1_MuOpen_DoubleEG5", 1),
  ("L1_ZeroBias_Instance1", 1),
  ("L1_DoubleJet44_Central", 1),
  ("L1_HTT50", 1),
  ("L1_DoubleEG_12_5_Eta1p39", 1),
  ("L1_TripleEG5", 1),
  ("L1_TripleEG_8_5_5", 1),
  ("L1_TripleEG_8_8_5", 1),
  ("L1_SingleJet80_Central", 1),
  ("L1_SingleJet92_Central", 1),
  ("L1_DoubleJet52", 1),
  ("L1_DoubleJet52_Central", 1),
  ("L1_TripleJet28_Central", 1),
  ("L1_TripleJet_36_36_12_Central", 1),
  ("L1_QuadJet20_Central", 1),
  ("L1_EG5_HTT75", 1),
  ("L1_EG5_HTT125", 1),
  ("L1_DoubleEG5_HTT50", 1)
#

  );
};

##########################################
#
# Only for experts:
# Select certain branches to speed up code.
# Modify only if you know what you do!
#
##########################################
branch:{
  doSelectBranches = false; #only set to true if you really know what you do!
  selectBranchL1 = true; 
  selectBranchHLT = true;
  selectBranchOpenHLT = true; 
  selectBranchReco = true;
  selectBranchL1extra = true; 
  selectBranchMC = false; 
};


### eof
