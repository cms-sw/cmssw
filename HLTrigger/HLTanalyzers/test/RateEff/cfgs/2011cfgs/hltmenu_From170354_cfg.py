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
    versionTag  = "20110721_DS_DoubleElectron"; 
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
 #lumiScaleFactor = 3.05; #run 170354, LS, to extrapolate to 3e33
 lumiScaleFactor = 5.09; #run 170354, LS, to extrapolate to 5e33
 prescaleNormalization = 1; 

##run 163374
runLumiblockList = ( 
   (170354, 5, 289 ) # (runnr, minLumiBlock, maxLumiBlock)
  );



};

##########################################
# Beam conditions
##########################################
beam:{
 bunchCrossingTime = 50.0E-09; # Design: 25 ns Startup: 75 ns
 iLumi = 5E33;
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

#castor, cern

## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__BTag_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__Commissioning_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__Cosmics_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__DoubleElectron_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__DoubleMu_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__ElectronHad_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__HT_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__Jet_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MinimumBias_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MET_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MuOnia_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MultiJet_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MuEG_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__MuHad_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__Photon_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__PhotonHad_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__SingleElectron_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__SingleMu_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__Tau_Run2011A-v1__20110719_0534/"];
## paths = ["dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/user/lpctrig/Commish2011/r170354__TauPlusX_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__BTag_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__Commissioning_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__Cosmics_Run2011A-v1__20110719_0534/"];
paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__DoubleElectron_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__DoubleMu_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__ElectronHad_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__HT_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__Jet_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MinimumBias_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MET_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MuOnia_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MultiJet_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MuEG_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__MuHad_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__Photon_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__PhotonHad_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__SingleElectron_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__SingleMu_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__Tau_Run2011A-v1__20110719_0534/"];
## paths = ["rfio:/castor/cern.ch/user/l/lucieg/OpenHLT/Commish2011/r170354__TauPlusX_Run2011A-v1__20110719_0534/"];



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

# preFilterByBits = "HLT_Photon50_CaloIdVL_v3";

  # (TriggerName, Prescale, EventSize)
 triggers = (
#
## ############# dataset BTag ###############
##    ("HLT_BTagMu_DiJet20_Mu5_v9", "L1_Mu3_Jet16_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet40_Mu5_v9", "L1_Mu3_Jet20_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet70_Mu5_v9", "L1_Mu3_Jet28_Central", 1, 0.15),
##    ("HLT_BTagMu_DiJet110_Mu5_v9", "L1_Mu3_Jet28_Central", 1, 0.15),
## ############# dataset Commissioning ###############
##    ("HLT_Activity_Ecal_SC7_v7", "L1_ZeroBias_Ext", 1, 0.15),
##    ("HLT_L1SingleJet16_v4", "L1_SingleJet16", 1, 0.15),
##    ("HLT_L1SingleJet36_v4", "L1_SingleJet36", 1, 0.15),
##    ("HLT_L1SingleMuOpen_v4", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_L1SingleMuOpen_DT_v4", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_Mu5_TkMu0_OST_Jpsi_Tight_B5Q7_v8", "L1_SingleMu5_Eta1p5_Q80", 1, 0.15),
##    ("HLT_L1SingleEG5_v3", "L1_SingleEG5", 1, 0.15),
##    ("HLT_L1SingleEG12_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_BeamGas_HF_v6", "L1_BeamGas_Hf", 1, 0.15),
##    ("HLT_BeamGas_BSC_v4", "L1_BeamGas_Bsc", 1, 0.15),
##    ("HLT_L1_PreCollisions_v3", "L1_PreCollisions", 1, 0.15),
##    ("HLT_L1_Interbunch_BSC_v3", "L1_InterBunch_Bsc", 1, 0.15),
##    ("HLT_IsoTrackHE_v7", "L1_SingleJet68", 1, 0.15),
##    ("HLT_IsoTrackHB_v6", "L1_SingleJet68", 1, 0.15),
## ############# dataset Cosmics ###############
##    ("HLT_BeamHalo_v5", "L1_BeamHalo", 1, 0.15),
##    ("HLT_L1SingleMuOpen_AntiBPTX_v3", "L1_SingleMuOpen", 1, 0.15),
##    ("HLT_L1TrackerCosmics_v4", "L1Tech_RPC_TTU_pointing_Cosmics.v0", 1, 0.15),
##    ("HLT_RegionalCosmicTracking_v6", "L1Tech_RPC_TTU_pointing_Cosmics.v0 AND L1_SingleMuOpen", 1, 0.15),
## ############# dataset DoubleElectron ###############
    ("HLT_Photon20_CaloIdVT_IsoT_Ele8_CaloIdL_CaloIsoVL_v7", "L1_DoubleEG_12_5", 7, 0.15),
    ("HLT_Ele8_v7", "L1_SingleEG5", 1, 0.15),
    ("HLT_Ele8_CaloIdL_CaloIsoVL_v7", "L1_SingleEG5", 1, 0.15),
    ("HLT_Ele8_CaloIdL_TrkIdVL_v7", "L1_SingleEG5", 1, 0.15),
    ("HLT_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v5", "L1_SingleEG5", 1, 0.15),
    ("HLT_Ele17_CaloIdL_CaloIsoVL_v7", "L1_SingleEG12", 1, 0.15),
#    ("HLT_Ele17_CaloIdL_CaloIsoVL_Ele8_CaloIdL_CaloIsoVL_v7", "L1_SingleEG12", 1, 0.15),
    ("HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC8_Mass30_v7", "L1_SingleEG12", 1, 0.15),
    ("HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v6", "L1_DoubleEG_12_5", 1, 0.15),
    ("HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele8_Mass30_v5", "L1_DoubleEG_12_5", 3, 0.15),
#    ("HLT_Ele17_CaloIdL_CaloIsoVL_Ele15_HFL_v8", "L1_EG12_ForJet16", 1, 0.15),
    ("OpenHLT_Ele22_CaloIdL_CaloIsoVL_Ele15_HFT", "OpenL1_EG18_ForJet16", 1, 0.15),
    ("HLT_Ele32_CaloIdT_CaloIsoT_TrkIdT_TrkIsoT_SC17_v5", "L1_SingleEG20", 3, 0.15),
    ("HLT_Ele8_CaloIdL_CaloIsoVL_Jet40_v7", "L1_SingleEG5", 1, 0.15),
    ("HLT_DoubleEle10_CaloIdL_TrkIdVL_Ele10_CaloIdT_TrkIdVL_v2", "L1_TripleEG5", 1, 0.15),
    ("HLT_TripleEle10_CaloIdL_TrkIdVL_v8", "L1_TripleEG5", 1, 0.15)
## ############# dataset DoubleMu ###############
##    ("HLT_L1DoubleMu0_v4", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_L2DoubleMu0_v7", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_L2DoubleMu23_NoVertex_v7", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_L2DoubleMu30_NoVertex_v3", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu3_v9", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_DoubleMu6_v7", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu7_v7", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu45_v5", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu4_Acoplanarity03_v8", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu5_Acoplanarity03_v5", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_Mu13_Mu8_v6", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_Mu17_Mu8_v6", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_TripleMu5_v8", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_DoubleMu5_IsoMu5_v7", "L1_DoubleMu3", 1, 0.15),
##    ("HLT_Mu8_Jet40_v9", "L1_Mu3_Jet20_Central", 1, 0.15),
## ############# dataset ElectronHad ###############
##    ("HLT_HT200_DoubleEle5_CaloIdVL_MassJPsi_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_v1", "L1_SingleEG12", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_v2", "L1_DoubleEG5", 1, 0.15),
##    ("HLT_HT300_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT40_v5", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_Ele5_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_PFMHT45_v5", "L1_HTT100", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_DiJet30_v4", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_TriJet30_v4", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele8_CaloIdT_TrkIdT_QuadJet30_v4", "L1_EG5_DoubleJet20_Central", 1, 0.15),
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_v7", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_Ele15_CaloIdT_CaloIsoVL_TrkIdT_TrkIsoVL_HT250_PFMHT25_v3", "L1_EG5_HTT100", 1, 0.15),
##    ("HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R005_MR200_v4", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R025_MR200_v4", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_Ele10_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_R029_MR200_v2", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_v7", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_DiCentralJet30_v6", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_TriCentralJet30_v6", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_QuadCentralJet30_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralJet30_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TriCentralJet30_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_QuadCentralJet30_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_BTagIP_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_TrkIdT_CentralJet30_BTagIP_v7", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele15_CaloIdVT_TrkIdT_Jet35_Jet25_Deta3_Jet20_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3_Jet20_v2", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele17_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_Jet35_Jet25_Deta3p5_Jet25_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele22_CaloIdVT_TrkIdT_CentralJet30_CentralJet25_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele22_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele22_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_CentralJet30_CentralJet25_PFMHT20_v4", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele25_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_DiCentralJet30_PFMHT25_v4", "L1_SingleEG12", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_HT150_v5", "L1_DoubleEG5_HTT50", 1, 0.15),
##    ("HLT_DoubleEle8_CaloIdT_TrkIdVL_Mass4_HT150_v2", "L1_DoubleEG5_HTT50", 1, 0.15),
## ############# dataset HT ###############
##    ("HLT_FatJetMass300_DR1p1_Deta2p0_CentralJet30_BTagIP_v2", "L1_HTT75", 1, 0.15),
##    ("HLT_FatJetMass350_DR1p1_Deta2p0_CentralJet30_BTagIP_v2", "L1_HTT75", 1, 0.15),
##    ("HLT_FatJetMass750_DR1p1_Deta2p0_v1", "L1_HTT75", 1, 0.15),
##    ("HLT_FatJetMass850_DR1p1_Deta2p0_v1", "L1_HTT75", 1, 0.15),
##    ("HLT_DiJet130_PT130_v6", "L1_SingleJet68", 1, 0.15),
##    ("HLT_DiJet160_PT160_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_HT150_v8", "L1_HTT50", 1, 0.15),
##    ("HLT_HT200_v8", "L1_HTT75", 1, 0.15),
##    ("HLT_HT200_AlphaT0p55_v2", "L1_HTT75", 1, 0.15),
##    ("HLT_HT250_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_AlphaT0p53_v6", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_AlphaT0p55_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_DoubleDisplacedJet60_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_DoubleDisplacedJet60_PromptTrack_v5", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_MHT90_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_MHT100_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_v9", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_MHT80_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_MHT90_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_PFMHT55_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_CentralJet30_BTagIP_v6", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_CentralJet30_BTagIP_PFMHT55_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_CentralJet30_BTagIP_PFMHT75_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_AlphaT0p53_v6", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_AlphaT0p54_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_MHT70_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_MHT80_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_AlphaT0p52_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_AlphaT0p53_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_AlphaT0p51_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT400_AlphaT0p52_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT450_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT450_AlphaT0p51_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT450_AlphaT0p52_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_HT500_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT500_JetPt60_DPhi2p94_v1", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_HT550_v8", "L1_HTT100", 1, 0.15),
##    ("HLT_HT550_JetPt60_DPhi2p94_v1", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_HT600_v1", "L1_HTT100", 1, 0.15),
##    ("HLT_HT2000_v2", "L1_HTT100", 1, 0.15),
##    ("HLT_R014_MR150_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R020_MR150_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R020_MR550_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R023_MR550_v2", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R025_MR150_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R025_MR450_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R029_MR450_v2", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R033_MR350_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R036_MR350_v2", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R038_MR250_v6", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_R042_MR250_v2", "L1_DoubleJet44_Central", 1, 0.15),
## ############# dataset Jet ###############
##    ("HLT_Jet30_v6", "L1_SingleJet16", 1, 0.15),
##    ("HLT_Jet60_v6", "L1_SingleJet36", 1, 0.15),
##    ("HLT_Jet80_v6", "L1_SingleJet52", 1, 0.15),
##    ("HLT_Jet110_v6", "L1_SingleJet68", 1, 0.15),
##    ("HLT_Jet150_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet190_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet240_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet240_CentralJet30_BTagIP_v2", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet270_CentralJet30_BTagIP_v2", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet300_v5", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet370_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet370_NoJetID_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_Jet800_v1", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve30_v6", "L1_SingleJet16", 1, 0.15),
##    ("HLT_DiJetAve60_v6", "L1_SingleJet36", 1, 0.15),
##    ("HLT_DiJetAve80_v6", "L1_SingleJet52", 1, 0.15),
##    ("HLT_DiJetAve110_v6", "L1_SingleJet68", 1, 0.15),
##    ("HLT_DiJetAve150_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve190_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve240_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve300_v6", "L1_SingleJet92", 1, 0.15),
##    ("HLT_DiJetAve370_v6", "L1_SingleJet92", 1, 0.15),
## ############# dataset MinimumBias ###############
##    ("HLT_JetE30_NoBPTX_v6", "L1_SingleJet20_Central_NotBptxOR", 1, 0.15),
##    ("HLT_JetE30_NoBPTX_NoHalo_v8", "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_JetE30_NoBPTX3BX_NoHalo_v8", "L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_JetE50_NoBPTX3BX_NoHalo_v3", "L1_SingleJet32_NotBptxOR_NotMuBeamHalo", 1, 0.15),
##    ("HLT_PixelTracks_Multiplicity80_v6", "L1_ETT220", 1, 0.15),
##    ("HLT_PixelTracks_Multiplicity100_v6", "L1_ETT220", 1, 0.15),
##    ("HLT_ZeroBias_v4", "L1_ZeroBias_Ext", 1, 0.15),
##    ("HLT_Physics_v2", "", 1, 0.15),
##    ("HLT_Random_v1", "", 1, 0.15),
## ############# dataset MET ###############
##    ("HLT_CentralJet80_MET65_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET80_v6", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET100_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_CentralJet80_MET160_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_DiJet60_MET45_v7", "L1_ETM20", 1, 0.15),
##    ("HLT_DiCentralJet20_MET80_v5", "L1_ETM30", 1, 0.15),
##    ("HLT_DiCentralJet20_BTagIP_MET65_v6", "L1_ETM30", 1, 0.15),
##    ("HLT_PFMHT150_v11", "L1_ETM30", 1, 0.15),
##    ("HLT_MET65_v4", "L1_ETM30", 1, 0.15),
##    ("HLT_MET65_HBHENoiseFiltered_v4", "L1_ETM30", 1, 0.15),
##    ("HLT_MET100_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_MET100_HBHENoiseFiltered_v5", "L1_ETM30", 1, 0.15),
##    ("HLT_MET120_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_MET120_HBHENoiseFiltered_v5", "L1_ETM30", 1, 0.15),
##    ("HLT_MET200_v7", "L1_ETM30", 1, 0.15),
##    ("HLT_MET200_HBHENoiseFiltered_v5", "L1_ETM30", 1, 0.15),
##    ("HLT_MET400_v2", "L1_ETM30", 1, 0.15),
##    ("HLT_L2Mu60_1Hit_MET40_v5", "L1_SingleMu20", 1, 0.15),
##    ("HLT_L2Mu60_1Hit_MET60_v5", "L1_SingleMu20", 1, 0.15),
## ############# dataset MuOnia ###############
##    ("HLT_DoubleMu3p5_Jpsi_Displaced_v2", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_DoubleMu4_LowMass_Displaced_v2", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_NoVertexing_v2", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Upsilon_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon4_Bs_Barrel_v7", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon5_Upsilon_Barrel_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon6_Bs_v6", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon7_Jpsi_X_Barrel_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon7_PsiPrime_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon10_Jpsi_Barrel_v5", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Jpsi_Muon_v6", "L1_TripleMu0", 1, 0.15),
##    ("HLT_Dimuon0_Upsilon_Muon_v6", "L1_TripleMu0", 1, 0.15),
##    ("HLT_Mu5_L2Mu2_Jpsi_v8", "L1_DoubleMu0", 1, 0.15),
##    ("HLT_Mu5_Track2_Jpsi_v8", "L1_SingleMu3", 1, 0.15),
##    ("HLT_Mu7_Track7_Jpsi_v9", "L1_SingleMu7", 1, 0.15),
## ############# dataset MultiJet ###############
##    ("HLT_DoubleJet30_ForwardBackward_v7", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet60_ForwardBackward_v7", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet70_ForwardBackward_v7", "L1_DoubleForJet32_EtaOpp", 1, 0.15),
##    ("HLT_DoubleJet80_ForwardBackward_v7", "L1_DoubleForJet44_EtaOpp", 1, 0.15),
##    ("HLT_CentralJet46_CentralJet38_DiBTagIP3D_v2", "L1_DoubleJet36_Central", 1, 0.15),
##    ("HLT_CentralJet60_CentralJet53_DiBTagIP3D_v1", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_QuadJet40_v7", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_QuadJet40_IsoPFTau40_v11", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_QuadJet45_IsoPFTau45_v6", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_QuadJet50_Jet40_Jet30_v3", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_QuadJet60_v6", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_QuadJet70_v6", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_EightJet120_v1", "L1_QuadJet20_Central", 1, 0.15),
##    ("HLT_ExclDiJet60_HFOR_v6", "L1_SingleJet36", 1, 0.15),
##    ("HLT_ExclDiJet60_HFAND_v6", "L1_SingleJet36_FwdVeto", 1, 0.15),
##    ("HLT_L1ETM30_v4", "L1_ETM30", 1, 0.15),
##    ("HLT_L1DoubleJet36Central_v4", "L1_DoubleJet36_Central", 1, 0.15),
##    ("HLT_L1MultiJet_v4", "L1_HTT50 OR L1_TripleJet28_Central OR L1_QuadJet20_Central", 1, 0.15),
## ############# dataset MuEG ###############
##    ("HLT_Mu3_Ele8_CaloIdT_CaloIsoVL_v2", "L1_Mu3_EG5", 1, 0.15),
##    ("HLT_Mu5_DoubleEle8_CaloIdT_TrkIdVL_v3", "L1_MuOpen_DoubleEG5", 1, 0.15),
##    ("HLT_Mu5_Ele8_CaloIdT_TrkIdVL_Ele8_CaloIdL_TrkIdVL_v3", "L1_MuOpen_DoubleEG5", 1, 0.15),
##    ("HLT_Mu8_Ele17_CaloIdL_v8", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v3", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu8_Photon20_CaloIdVT_IsoT_v8", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu15_Photon20_CaloIdL_v9", "L1_MuOpen_EG12", 1, 0.15),
##    ("HLT_Mu15_DoublePhoton15_CaloIdL_v9", "L1_Mu7_EG5", 1, 0.15),
##    ("HLT_Mu17_Ele8_CaloIdL_v8", "L1_Mu7_EG5", 1, 0.15),
##    ("HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_v3", "L1_Mu7_EG5", 1, 0.15),
##    ("HLT_DoubleMu5_Ele8_CaloIdL_TrkIdVL_v9", "L1_DoubleMuOpen_EG5", 1, 0.15),
##    ("HLT_DoubleMu5_Ele8_CaloIdT_TrkIdVL_v3", "L1_DoubleMuOpen_EG5", 1, 0.15),
## ############# dataset MuHad ###############
##    ("HLT_Mu3_Ele8_CaloIdT_TrkIdVL_HT150_v6", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu8_R005_MR200_v7", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_Mu8_R025_MR200_v7", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_Mu8_R029_MR200_v3", "L1_DoubleJet44_Central", 1, 0.15),
##    ("HLT_HT250_Mu15_PFMHT20_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT250_Mu15_PFMHT40_v3", "L1_HTT100", 1, 0.15),
##    ("HLT_HT300_Mu5_PFMHT40_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_HT350_Mu5_PFMHT45_v7", "L1_HTT100", 1, 0.15),
##    ("HLT_Mu3_DiJet30_v6", "L1_Mu3_Jet20_Central", 1, 0.15),
##    ("HLT_Mu3_TriJet30_v6", "L1_Mu3_Jet20_Central", 1, 0.15),
##    ("HLT_Mu3_QuadJet30_v6", "L1_Mu3_Jet20_Central", 1, 0.15),
##    ("HLT_Mu17_CentralJet30_v10", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu17_DiCentralJet30_v10", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu17_TriCentralJet30_v10", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu17_QuadCentralJet30_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu12_DiCentralJet30_BTagIP3D_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu12_DiCentralJet20_BTagIP3D1stTrack_v2", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu12_DiCentralJet20_DiBTagIP3D1stTrack_v3", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu17_CentralJet30_BTagIP_v9", "L1_SingleMu10", 1, 0.15),
##    ("HLT_Mu30_HT200_v3", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_Mu40_HT200_v3", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_IsoMu17_CentralJet30_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu17_DiCentralJet30_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu17_TriCentralJet30_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu17_QuadCentralJet30_v5", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu17_CentralJet30_BTagIP_v9", "L1_SingleMu10", 1, 0.15),
##    ("HLT_IsoMu20_DiCentralJet34_v2", "L1_SingleMu10", 1, 0.15),
##    ("HLT_DoubleMu3_HT150_v7", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_DoubleMu3_Mass4_HT150_v3", "L1_Mu0_HTT50", 1, 0.15),
##    ("HLT_DoubleMu3_HT200_v10", "L1_Mu0_HTT50", 1, 0.15),
############# dataset Photon ###############
##   ("HLT_Photon20_CaloIdVL_IsoL_v6", "L1_SingleEG12", 1, 0.15),
##   ("HLT_Photon20_R9Id_Photon18_R9Id_v6", "L1_SingleEG12", 1, 0.15),
##   ("HLT_Photon26_Photon18_v6", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_IsoVL_Photon18_v7", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_IsoVL_Photon18_IsoVL_v7", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_CaloIdL_IsoVL_Photon18_v7", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_CaloIdL_IsoVL_Photon18_R9Id_v6", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_CaloIdL_IsoVL_Photon18_CaloIdL_IsoVL_v7", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_R9Id_Photon18_CaloIdL_IsoVL_v6", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon26_R9Id_Photon18_R9Id_v3", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon30_CaloIdVL_v6", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon30_CaloIdVL_IsoL_v7", "L1_SingleEG15", 1, 0.15),
##   ("HLT_Photon36_IsoVL_Photon22_v4", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdVL_Photon22_CaloIdVL_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdL_Photon22_CaloIdL_v5", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdL_IsoVL_Photon22_v4", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdL_IsoVL_Photon22_CaloIdL_IsoVL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_CaloIdL_IsoVL_Photon22_R9Id_v2", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_R9Id_Photon22_CaloIdL_IsoVL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon36_R9Id_Photon22_R9Id_v2", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon40_CaloIdL_Photon28_CaloIdL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon44_CaloIdL_Photon34_CaloIdL_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon48_CaloIdL_Photon38_CaloIdL_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon50_CaloIdVL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon50_CaloIdVL_IsoL_v6", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon75_CaloIdVL_v6", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon75_CaloIdVL_IsoL_v7", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon90_CaloIdVL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon90_CaloIdVL_IsoL_v4", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon135_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon225_NoHE_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon400_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_Photon200_NoHE_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton33_HEVT_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton38_HEVT_v2", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton60_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton80_v1", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton5_IsoVL_CEP_v6", "L1_DoubleEG2_FwdVeto", 1, 0.15),
##   ("HLT_DoubleEle33_v4", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoubleEle33_CaloIdL_v4", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoubleEle45_CaloIdL_v3", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton40_MR150_v5", "L1_SingleEG20", 1, 0.15),
##   ("HLT_DoublePhoton40_R014_MR150_v5", "L1_SingleEG20", 1, 0.15)#,
## ############# dataset PhotonHad ###############
##    ("HLT_Photon70_CaloIdL_HT350_v7", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdL_HT400_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdL_MHT70_v7", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon70_CaloIdL_MHT90_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R005_MR150_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R014_MR500_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R017_MR500_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R020_MR350_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R023_MR350_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R025_MR250_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R029_MR250_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R038_MR200_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon40_CaloIdL_R042_MR200_v2", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Photon30_CaloIdVT_CentralJet20_BTagIP_v1", "L1_SingleEG15", 1, 0.15),
## ############# dataset SingleElectron ###############
##    ("HLT_Ele25_CaloIdL_CaloIsoVL_TrkIdVL_TrkIsoVL_v3", "L1_SingleEG12", 1, 0.15),
##    ("HLT_Ele27_WP80_PFMT50_v3", "L1_SingleEG15", 1, 0.15),
##    ("HLT_Ele32_WP70_PFMT50_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele32_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v6", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele52_CaloIdVT_TrkIdT_v4", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele65_CaloIdVT_TrkIdT_v3", "L1_SingleEG20", 1, 0.15),
##    ("HLT_Ele100_CaloIdVL_CaloIsoVL_TrkIdVL_TrkIsoVL_v2", "L1_SingleEG20", 1, 0.15),
## ############# dataset SingleMu ###############
##   ("HLT_L1SingleMu10_v4", "L1_SingleMu10", 1, 0.15),
##   ("HLT_L1SingleMu20_v4", "L1_SingleMu20", 1, 0.15),
##   ("HLT_L2Mu10_v6", "L1_SingleMu10", 1, 0.15),
##   ("HLT_L2Mu20_v6", "L1_SingleMu12", 1, 0.15),
##   ("HLT_Mu3_v9", "L1_SingleMuOpen", 1, 0.15),
##   ("HLT_Mu5_v9", "L1_SingleMu3", 1, 0.15),
##   ("HLT_Mu8_v7", "L1_SingleMu3", 1, 0.15),
##   ("HLT_Mu12_v7", "L1_SingleMu7", 1, 0.15),
##   ("HLT_Mu15_v8", "L1_SingleMu10", 1, 0.15),
##   ("HLT_Mu20_v7", "L1_SingleMu12", 1, 0.15),
##   ("HLT_Mu24_v7", "L1_SingleMu12", 1, 0.15),
##   ("HLT_Mu30_v7", "L1_SingleMu12", 2, 0.15),
##   ("HLT_Mu40_v5", "OpenL1_SingleMu14_Eta2p1", 1, 0.15),
##   ("HLT_Mu60_v2", "OpenL1_SingleMu14_Eta2p1", 1, 0.15),
##   ("HLT_Mu100_v5", "OpenL1_SingleMu14_Eta2p1", 1, 0.15),
##   ("HLT_IsoMu12_v9", "L1_SingleMu7", 1, 0.15),
##   ("HLT_IsoMu15_v13", "L1_SingleMu10", 1, 0.15),
##   ("HLT_IsoMu17_v13", "L1_SingleMu10", 2, 0.15),
##   ("HLT_IsoMu20_v8", "L1_SingleMu12", 2, 0.15),
##   ("HLT_IsoMu24_v8", "L1_SingleMu12", 2, 0.15),
##   ("HLT_IsoMu30_v8", "OpenL1_SingleMu14_Eta2p1", 1, 0.15),
##   ("OpenHLT_IsoMu34", "OpenL1_SingleMu14_Eta2p1", 1, 0.15)
## ############# dataset Tau ###############
##   ("HLT_IsoPFTau35_Trk20_v6", "L1_SingleJet52_Central", 10, 0.15),
##   ("HLT_IsoPFTau35_Trk20_MET60_v6", "L1_Jet52_Central_ETM30", 30, 0.15),
##   ("HLT_IsoPFTau35_Trk20_MET70_v2", "L1_Jet52_Central_ETM30", 1, 0.15),
##   ("HLT_DoubleIsoPFTau45_Trk5_eta2p1_v2", "L1_DoubleTauJet40_Eta2p17 OR L1_DoubleJet52_Central", 1, 0.15),
##   ("HLT_IsoPFTau40_IsoPFTau30_Trk5_eta2p1_v2", "L1_DoubleTauJet36_Eta2p17 OR L1_DoubleJet44_Central", 40, 0.15)
## ############# dataset TauPlusX ###############
##   ("HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v7", "L1_SingleEG12", 1, 0.15),
##   ("HLT_HT300_DoubleIsoPFTau10_Trk3_PFMHT40_v7", "L1_HTT100", 1, 0.15),
##   ("HLT_HT350_DoubleIsoPFTau10_Trk3_PFMHT45_v7", "L1_HTT100", 1, 0.15),
##   ("HLT_Mu15_LooseIsoPFTau15_v8", "L1_SingleMu10", 1, 0.15),
##   ("HLT_IsoMu15_LooseIsoPFTau15_v8", "L1_SingleMu10", 1, 0.15),
##   ("HLT_IsoMu15_LooseIsoPFTau20_v6", "L1_SingleMu10", 1, 0.15),
##   ("HLT_IsoMu15_TightIsoPFTau20_v6", "L1_SingleMu10", 1, 0.15),
##   ("HLT_Ele15_CaloIdVT_TrkIdT_TightIsoPFTau20_v2", "L1_SingleEG12", 1, 0.15),
##   ("HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v2", "L1_SingleEG12", 1, 0.15),
##   ("HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v2", "L1_SingleEG15", 1, 0.15)
# 
 );

 # For L1 prescale preloop to be used in HLT mode only
 L1triggers = ( 
#
  ("L1_SingleMu7", 100),
  ("L1_SingleMu10", 80),
  ("L1_SingleMu12", 60),
  ("L1_SingleMu16", 20),
  ("L1_SingleMu20", 20),
  ("OpenL1_SingleMu14_Eta2p1", 1),
  ("OpenL1_SingleMu16_Eta2p1", 1),
  ("L1_SingleJet20_Central_NotBptxOR_NotMuBeamHalo", 1),
  ("L1_EG15_ForJet16", 1),
  ("OpenL1_EG18_ForJet16", 1),
  ("L1_SingleJet36_FwdVeto", 1),
  ("L1_MuOpen_EG12", 1),
  ("L1_BeamGas_Bsc", 1),
  ("L1_ETT220", 1),
  ("L1_SingleEG5", 10),
  ("L1_EG5_HTT100", 1),
  ("L1_SingleJet16", 1),
  ("L1_Mu3_Jet20_Central", 1),
  ("L1_HTT50", 1),
  ("L1_TripleJet28_Central", 1),
  ("L1_QuadJet20_Central", 1),
  ("L1_DoubleForJet32_EtaOpp", 1),
  ("L1_SingleMuOpen", 7),
  ("L1_TripleEG5", 1),
  ("L1_SingleJet52_Central", 1),
  ("L1_HTT75", 1),
  ("L1Tech_HCAL_HF_MM_or_PP_or_PM.v0", 1),
  ("L1_SingleJet52", 1),
  ("L1_ETM30", 1),
  ("L1_BeamHalo", 1),
  ("L1_DoubleTauJet40_Eta2p17", 1),
  ("L1_DoubleJet52_Central", 1),
  ("L1_Jet52_Central_ETM30", 1),
  ("L1_EG5_DoubleJet20_Central", 1),
  ("L1_ETM20", 1),
  ("L1_Mu7_EG5", 1),
  ("L1_SingleJet36", 1),
  ("L1_SingleJet68", 1),
  ("L1_SingleJet92", 1),
  ("L1_SingleJet128", 1),
  ("L1_SingleMu3", 14),
  ("L1_SingleIsoEG12", 1),
  ("L1_SingleEG12", 100),
  ("L1_SingleEG15", 1),
  ("L1_SingleEG20", 1),
  ("L1_SingleEG30", 1),
  ("L1_ZeroBias_Ext", 1),
  ("L1_DoubleTauJet36_Eta2p17", 1),
  ("L1_DoubleJet44_Central", 1),
  ("L1Tech_HCAL_HO_totalOR.v0", 1),
  ("L1Tech_HCAL_HBHE_totalOR.v0", 1),
  ("L1_InterBunch_Bsc", 1),
  ("L1_Mu3_EG5", 1),
  ("L1_Mu3_Jet16_Central", 1),
  ("L1Tech_RPC_TTU_pointing_Cosmics.v0", 1),
  ("L1_Mu3_Jet28_Central", 1),
  ("L1_PreCollisions", 1),
  ("L1_SingleJet32_NotBptxOR_NotMuBeamHalo", 1),
  ("L1_EG12_ForJet16", 1),
  ("L1_DoubleMu0", 1),
  ("L1_DoubleMu3", 1),
  ("L1_SingleMu5_Eta1p5_Q80", 1),
  ("L1_HTT100", 1),
  ("L1_DoubleEG5_HTT50", 1),
  ("L1_DoubleJet36_Central", 1),
  ("L1_SingleJet20_Central_NotBptxOR", 1),
  ("L1_DoubleForJet44_EtaOpp", 1),
  ("L1_DoubleEG2_FwdVeto", 1),
  ("L1_TripleMu0", 1),
  ("L1_DoubleMuOpen_EG5", 1),
  ("L1_DoubleEG5", 1),
  ("L1_DoubleEG10", 1),
  ("L1_DoubleEG3", 1),
  ("L1_DoubleEG8", 1),
  ("L1_DoubleEG_12_5", 1),
  ("L1_DoubleIsoEG10", 1),
  ("L1_SingleEG12_Eta1p39", 1),
  ("L1_SingleIsoEG12_Eta2p17", 1),
  ("L1_SingleMu25", 1),
  ("L1_DoubleMu5", 1),
  ("L1_BeamGas_Hf", 1),
  ("L1_MuOpen_DoubleEG5", 1),
  ("L1_Mu0_HTT50", 1),
  ("L1_DoubleEG_12_5_Eta1p39", 1),
  ("L1_TripleEG7", 1),
  ("L1_TripleEG_8_5_5", 1),
  ("L1_TripleEG_8_8_5", 1),
  ("L1_SingleIsoEG12_Eta1p39", 1),
  ("L1_SingleJet80_Central", 1),
  ("L1_SingleJet92_Central", 1),
  ("L1_DoubleJet52", 1),
  ("L1_TripleJet_36_36_12_Central", 1),
  ("L1_QuadJet28_Central", 1),
  ("L1_EG5_HTT75", 1),
  ("L1_EG5_HTT125", 1),
  ("L1_DoubleEG5_HTT75", 1)
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
  doSelectBranches = true; #only set to true if you really know what you do!
  selectBranchL1 = true; 
  selectBranchHLT = true;
  selectBranchOpenHLT = true; 
  selectBranchReco = true;
  selectBranchL1extra = true; 
  selectBranchMC = false; 
};


### eof
