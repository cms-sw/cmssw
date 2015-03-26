#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Common/interface/TriggerNames.h" 
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

HcalRecHitMonitor::HcalRecHitMonitor(const edm::ParameterSet& ps):HcalBaseDQMonitor(ps)
{
  // Common Base Class parameters
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","RecHitMonitor_Hcal/"); 
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  hbheRechitLabel_       = ps.getUntrackedParameter<edm::InputTag>("hbheRechitLabel");
  hoRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hoRechitLabel");
  hfRechitLabel_         = ps.getUntrackedParameter<edm::InputTag>("hfRechitLabel");
  l1gtLabel_             = ps.getUntrackedParameter<edm::InputTag>("L1GTLabel"); // should be l1GtUnpack

  hltresultsLabel_       = ps.getUntrackedParameter<edm::InputTag>("HLTResultsLabel");
  HcalHLTBits_           = ps.getUntrackedParameter<std::vector<std::string> >("HcalHLTBits");
  MinBiasHLTBits_        = ps.getUntrackedParameter<std::vector<std::string> >("MinBiasHLTBits");

  // energy/ET threshold plots also require that at least one MinBias trigger bit fires
  energyThreshold_       = ps.getUntrackedParameter<double>("energyThreshold",2);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("HF_energyThreshold",energyThreshold_);
  
  ETThreshold_       = ps.getUntrackedParameter<double>("ETThreshold",0);
  HBETThreshold_     = ps.getUntrackedParameter<double>("HB_ETThreshold",ETThreshold_);
  HEETThreshold_     = ps.getUntrackedParameter<double>("HE_ETThreshold",ETThreshold_);
  HOETThreshold_     = ps.getUntrackedParameter<double>("HO_ETThreshold",ETThreshold_);
  HFETThreshold_     = ps.getUntrackedParameter<double>("HF_ETThreshold",ETThreshold_);

  timediffThresh_    = ps.getUntrackedParameter<double>("collisiontimediffThresh",10.);
  setupDone_         = false;
  needLogicalMap_    = true;

  // register for data access
  tok_hbhe_ = consumes<HBHERecHitCollection>(hbheRechitLabel_);
  tok_ho_ = consumes<HORecHitCollection>(hoRechitLabel_);
  tok_hf_ = consumes<HFRecHitCollection>(hfRechitLabel_);
  tok_trigger_ = consumes<edm::TriggerResults>(hltresultsLabel_);

} //constructor

HcalRecHitMonitor::~HcalRecHitMonitor()
{
} //destructor


/* ------------------------------------ */ 


void HcalRecHitMonitor::setup(DQMStore::IBooker &ib)
{

  if (setupDone_)
  {
    // Always do a zeroing/resetting so that empty histograms/counter
    // will always appear.
  
    // clear all counters, reset histograms
    this->zeroCounters();
    this->reset();
    return;
  }
  else
    setupDone_=true;

  HcalBaseDQMonitor::setup(ib);


  if (debug_>0)
    std::cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<std::endl;

  // RecHit Monitor - specific cfg variables

  if (debug_>1)
    std::cout <<"<HcalRecHitMonitor::setup>  Creating Histograms"<<std::endl;

  ib.setCurrentFolder(subdir_);
  h_TriggeredEvents=ib.book1D("EventTriggers","EventTriggers",3,-0.5,2.5);
  h_TriggeredEvents->setBinLabel(1,"AllEvents");
  h_TriggeredEvents->setBinLabel(2,"HLT_Minbias");
  h_TriggeredEvents->setBinLabel(3,"HLT_Hcal");

  ib.setCurrentFolder(subdir_+"rechit_parameters");
  MonitorElement* THR;
  ib.setCurrentFolder(subdir_+"rechit_parameters/thresholds");
  THR=ib.bookFloat("HB_Rechit_Energy_Threshold");
  THR->Fill(HBenergyThreshold_);
  THR=ib.bookFloat("HE_Rechit_Energy_Threshold");
  THR->Fill(HEenergyThreshold_);
  THR=ib.bookFloat("HO_Rechit_Energy_Threshold");
  THR->Fill(HOenergyThreshold_);
  THR=ib.bookFloat("HF_Rechit_Energy_Threshold");
  THR->Fill(HFenergyThreshold_);
  THR=ib.bookFloat("HB_Rechit_ET_Threshold");
  THR->Fill(HBETThreshold_);
  THR=ib.bookFloat("HE_Rechit_ET_Threshold");
  THR->Fill(HEETThreshold_);
  THR=ib.bookFloat("HO_Rechit_ET_Threshold");
  THR->Fill(HOETThreshold_);
  THR=ib.bookFloat("HF_Rechit_ET_Threshold");
  THR->Fill(HFETThreshold_);
  THR=ib.bookFloat("Maximum_HFM_HFP_time_difference_for_luminosityplots");
  THR->Fill(timediffThresh_);

  
  // Set up histograms that are filled by all rechits
  ib.setCurrentFolder(subdir_+"Distributions_AllRecHits");
  SetupEtaPhiHists(ib,OccupancyByDepth,"RecHit Occupancy","");
  h_rechitieta = ib.book1D("HcalRecHitIeta",
			      "Hcal RecHit ieta",
			      83,-41.5,41.5);
  h_rechitiphi = ib.book1D("HcalRecHitIphi",
			      "Hcal RecHit iphi",
			      72,0.5,72.5);

  h_rechitieta_05 = ib.book1D("HcalRecHitIeta05",
				 "Hcal RecHit ieta E>0.5 GeV",
				 83,-41.5,41.5);
  h_rechitiphi_05 = ib.book1D("HcalRecHitIphi05",
				 "Hcal RecHit iphi E>0.5 GeV",
				 72,0.5,72.5);
  h_rechitieta_10 = ib.book1D("HcalRecHitIeta10",
				 "Hcal RecHit ieta E>1.0 GeV",
				 83,-41.5,41.5);
  h_rechitiphi_10 = ib.book1D("HcalRecHitIphi10",
				 "Hcal RecHit iphi E>1.0 GeV",
				 72,0.5,72.5);
  h_rechitieta_25 = ib.book1D("HcalRecHitIeta25",
				 "Hcal RecHit ieta E>2.5 GeV",
				 83,-41.5,41.5);
  h_rechitiphi_25 = ib.book1D("HcalRecHitIphi25",
				 "Hcal RecHit iphi E>2.5 GeV",
				 72,0.5,72.5);
  h_rechitieta_100 = ib.book1D("HcalRecHitIeta100",
				  "Hcal RecHit ieta E>10.0 GeV",
				  83,-41.5,41.5);
  h_rechitiphi_100 = ib.book1D("HcalRecHitIphi100",
				  "Hcal RecHit iphi E>10.0 GeV",
				  72,0.5,72.5);



  h_LumiPlot_LS_allevents = ib.book1D("AllEventsPerLS",
					 "LS # of all events",
					 NLumiBlocks_,0.5,NLumiBlocks_+0.5);
  h_LumiPlot_BX_allevents = ib.book1D("BX_allevents",
					 "BX # of all events",
					 3600,0,3600);
  h_LumiPlot_MinTime_vs_MinHT = ib.book2D("MinTime_vs_MinSumET",
					     "Energy-Weighted Time vs Min (HF+,HF-) Scalar Sum ET;min Sum ET(GeV);time(ns)",
					     100,0,10,80,-40,40);

  h_LumiPlot_timeHT_HFM = ib.book2D("HFM_Time_vs_SumET",
				       "Energy-Weighted Time vs HFMinus Scalar Sum ET;Sum ET(GeV);time(ns)",
				       100,0,10,80,-40,40);
  
  h_LumiPlot_timeHT_HFP = ib.book2D("HFP_Time_vs_SumET",
				       "Energy-Weighted Time vs HFPlus Scalar Sum ET;Sum ET(GeV);time(ns)",
				       100,0,10,80,-40,40);


  ib.setCurrentFolder(subdir_+"Distributions_AllRecHits/sumplots");
  SetupEtaPhiHists(ib,SumEnergyByDepth,"RecHit Summed Energy","GeV");
  SetupEtaPhiHists(ib,SqrtSumEnergy2ByDepth,"RecHit Sqrt Summed Energy2","GeV");
  SetupEtaPhiHists(ib,SumTimeByDepth,"RecHit Summed Time","nS");

  // Histograms for events that passed MinBias triggers
  ib.setCurrentFolder(subdir_+"Distributions_PassedMinBias");

  h_HBP_weightedTime = ib.book1D("WeightedTime_HBP","Weighted Time for HBP",
				    300,-150,150);
  h_HBM_weightedTime = ib.book1D("WeightedTime_HBM","Weighted Time for HBM",
				    300,-150,150);
  h_HEP_weightedTime = ib.book1D("WeightedTime_HEP","Weighted Time for HEP",
				    300,-150,150);
  h_HEM_weightedTime = ib.book1D("WeightedTime_HEM","Weighted Time for HEM",
				    300,-150,150);
  h_HFP_weightedTime = ib.book1D("WeightedTime_HFP","Weighted Time for HFP",
				    300,-150,150);
  h_HFM_weightedTime = ib.book1D("WeightedTime_HFM","Weighted Time for HFM",
				    300,-150,150);

  h_HFtimedifference = ib.book1D("HFweightedtimeDifference",
				    "Energy-Weighted time difference between HF+ and HF- passing MinBias (no HT cut)",
				    251,-250.5,250.5);
  h_HEtimedifference = ib.book1D("HEweightedtimeDifference",
				    "Energy-Weighted time difference between HE+ and HE- passing MinBias (no HT cut)",
				    251,-250.5,250.5);
  
  HFP_HFM_Energy = ib.book2D("HFP_HFM_Energy",
				"HFP VS HFM Energy; Total Energy in HFMinus (TeV); Total Energy in HFPlus (TeV)",
				100,0,100, 100,0,100);
  
  // Would these work better as 2D plots?
  h_HFenergydifference = ib.book1D("HFenergyDifference",
				      "Sum(E_HFPlus - E_HFMinus)/Sum(E_HFPlus + E_HFMinus)",
				      200,-1,1);
  h_HEenergydifference = ib.book1D("HEenergyDifference",
				      "Sum(E_HEPlus - E_HEMinus)/Sum(E_HEPlus + E_HEMinus)",
				      200,-1,1);

  h_LumiPlot_LS_MinBiasEvents=ib.book1D("MinBiasEventsPerLS",
					   "Number of MinBias Events vs LS (HT cut and HFM-HFP time cut)",
					   NLumiBlocks_/10,0.5,NLumiBlocks_+0.5); 
  h_LumiPlot_LS_MinBiasEvents_notimecut=ib.book1D("MinBiasEventsPerLS_notimecut",
						     "Number of Events with MinBias vs LS (HFM,HFP HT>1,no time cut)",
						     NLumiBlocks_/10,0.5,NLumiBlocks_+0.5); 

  h_LumiPlot_SumHT_HFPlus_vs_HFMinus = ib.book2D("SumHT_plus_minus",
						    "HF+ Sum HT vs HF- Sum HT",60,0,30,60,0,30);
  h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus = ib.book2D("SumEnergy_plus_minus",
							"HF+ Sum Energy  vs HF- Sum Energy",
							60,0,150,60,0,150);
  h_LumiPlot_timeHFPlus_vs_timeHFMinus = ib.book2D("timeHFplus_vs_timeHFminus",
						      "Energy-weighted time average of HF+ vs HF-",
						      60,-60,60,60,-60,60);
  h_LumiPlot_BX_MinBiasEvents = ib.book1D("BX_MinBias_Events_TimeCut",
					  "BX # of MinBias events (HFM & HFP HT>1 & HFM-HFP time cut)",
					  3600,0,3600);
  h_LumiPlot_BX_MinBiasEvents_notimecut = ib.book1D("BX_MinBias_Events_notimecut",
						       "BX # of MinBias events (HFM,HFP HT>1, no time cut)",
						       3600,0,3600);
  // threshold plots must pass MinBias Trigger
  SetupEtaPhiHists(ib,OccupancyThreshByDepth,"Above Threshold RecHit Occupancy","");
  h_rechitieta_thresh = ib.book1D("HcalRecHitIeta_thresh",
			      "Hcal RecHit ieta above energy and ET threshold",
			      83,-41.5,41.5);
  h_rechitiphi_thresh = ib.book1D("HcalRecHitIphi_thresh",
			      "Hcal RecHit iphi above energy and ET threshold",
			      72,0.5,72.5);

  ib.setCurrentFolder(subdir_+"Distributions_PassedMinBias/sumplots");
  SetupEtaPhiHists(ib,SumEnergyThreshByDepth,"Above Threshold RecHit Summed Energy","GeV");
  SetupEtaPhiHists(ib,SumTimeThreshByDepth,"Above Threshold RecHit Summed Time","nS");
  SetupEtaPhiHists(ib,SqrtSumEnergy2ThreshByDepth,"Above Threshold RecHit Sqrt Summed Energy2","GeV");

  ib.setCurrentFolder(subdir_+"Distributions_PassedMinBias/rechit_1D_plots");
  h_HBThreshTime=ib.book1D("HB_time_thresh", 
			      "HB RecHit Time Above Threshold",
			      int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HBThreshOccupancy=ib.book1D("HB_occupancy_thresh",
				   "HB RecHit Occupancy Above Threshold",260,-0.5,2599.5);
  h_HEThreshTime=ib.book1D("HE_time_thresh", 
			      "HE RecHit Time Above Threshold",
			      int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HEThreshOccupancy=ib.book1D("HE_occupancy_thresh",
				   "HE RecHit Occupancy Above Threshold",260,-0.5,2599.5);
  h_HOThreshTime=ib.book1D("HO_time_thresh", 
			      "HO RecHit Time Above Threshold",
			      int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HOThreshOccupancy=ib.book1D("HO_occupancy_thresh",
				   "HO RecHit Occupancy Above Threshold",217,-0.5,2169.5);
  h_HFThreshTime=ib.book1D("HF_time_thresh", 
			      "HF RecHit Time Above Threshold",
			      int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HFThreshOccupancy=ib.book1D("HF_occupancy_thresh",
				   "HF RecHit Occupancy Above Threshold",
				   173,-0.5,1729.5);

  // Histograms for events that did passed Hcal-specfied HLT triggers
  ib.setCurrentFolder(subdir_+"Distributions_PassedHcalHLTriggers");
  
  h_LumiPlot_BX_HcalHLTEvents = ib.book1D("BX_HcalHLT_Events_TimeCut",
					  "BX # of HcalHLT events (HFM & HFP HT>1 & HFM-HFP time cut)",
					  3600,0,3600);
  h_LumiPlot_BX_HcalHLTEvents_notimecut = ib.book1D("BX_HcalHLT_Events_notimecut",
						       "BX # of HcalHLT events (HFM,HFP HT>1, no time cut)",
						       3600,0,3600);
  h_LumiPlot_LS_HcalHLTEvents=ib.book1D("HcalHLTEventsPerLS",
					   "Number of HcalHLT Events vs LS (HT cut and HFM-HFP time cut)",
					   NLumiBlocks_/10,0.5,NLumiBlocks_+0.5); 
  h_LumiPlot_LS_HcalHLTEvents_notimecut=ib.book1D("HcalHLTEventsPerLS_notimecut",
						     "Number of Events with HcalHLT vs LS (HFM,HFP HT>1,no time cut)",
						     NLumiBlocks_/10,0.5,NLumiBlocks_+0.5); 
  

  ib.setCurrentFolder(subdir_+"Distributions_PassedHcalHLTriggers/");
  h_HF_HcalHLT_weightedtimedifference = ib.book1D("HF_HcalHLT_weightedtimeDifference",
					     "Energy-Weighted time difference between HF+ and HF- Hcal HLT",
					     251,-250.5,250.5);
  h_HE_HcalHLT_weightedtimedifference = ib.book1D("HE_HcalHLT_weightedtimeDifference",
					     "Energy-Weighted time difference between HE+ and HE- Hcal HLT",
					     251,-250.5,250.5);
  h_HF_HcalHLT_energydifference = ib.book1D("HF_HcalHLT_energyDifference",
					     "Sum(E_HFPlus - E_HFMinus)/Sum(E_HFPlus + E_HFMinus)",
					     200,-1,1);
  h_HE_HcalHLT_energydifference = ib.book1D("HE_HcalHLT_energyDifference",
					     "Sum(E_HEPlus - E_HEMinus)/Sum(E_HEPlus + E_HEMinus)",
					     200,-1,1);

  // Do we want separate directories for Minbias, other flags at some point?
  ib.setCurrentFolder(subdir_+"AnomalousCellFlags");// HB Flag Histograms


  h_HFLongShort_vs_LS=ib.book1D("HFLongShort_vs_LS",
				   "HFLongShort Flags vs Lumi Section",
				   NLumiBlocks_/10,0.5,0.5+NLumiBlocks_);
  h_HFDigiTime_vs_LS=ib.book1D("HFDigiTime_vs_LS",
				  "HFDigiTime Flags vs Lumi Section",
				  NLumiBlocks_/10,0.5,0.5+NLumiBlocks_);
  h_HBHEHPDMult_vs_LS=ib.book1D("HBHEHPDMult_vs_LS",
				   "HBHEHPDMult Flags vs Lumi Section",
				   NLumiBlocks_/10,0.5,0.5+NLumiBlocks_);
  h_HBHEPulseShape_vs_LS=ib.book1D("HBHEPulseShape_vs_LS",
				      "HBHEPulseShape Flags vs Lumi Section",
				      NLumiBlocks_/10,0.5,0.5+NLumiBlocks_);

  h_HF_FlagCorr=ib.book2D("HF_FlagCorrelation",
			     "HF LongShort vs. DigiTime flags; DigiTime; LongShort", 
			     2,-0.5,1.5,2,-0.5,1.5);
  h_HF_FlagCorr->setBinLabel(1,"OFF",1);
  h_HF_FlagCorr->setBinLabel(2,"ON",1);
  h_HF_FlagCorr->setBinLabel(1,"OFF",2);
  h_HF_FlagCorr->setBinLabel(2,"ON",2);

  h_HBHE_FlagCorr=ib.book2D("HBHE_FlagCorrelation",
			       "HBHE HpdHitMultiplicity vs. PulseShape flags; PulseShape; HpdHitMultiplicity", 
			       2,-0.5,1.5,2,-0.5,1.5);
  h_HBHE_FlagCorr->setBinLabel(1,"OFF",1);
  h_HBHE_FlagCorr->setBinLabel(2,"ON",1);
  h_HBHE_FlagCorr->setBinLabel(1,"OFF",2);
  h_HBHE_FlagCorr->setBinLabel(2,"ON",2);

  h_FlagMap_HPDMULT=ib.book2D("FlagMap_HPDMULT",
				 "RBX Map of HBHEHpdHitMultiplicity Flags;RBX;RM",
				  72,-0.5,71.5,4,0.5,4.5);
  h_FlagMap_PULSESHAPE=ib.book2D("FlagMap_PULSESHAPE",
				    "RBX Map of HBHEPulseShape Flags;RBX;RM",
				  72,-0.5,71.5,4,0.5,4.5);
  h_FlagMap_DIGITIME=ib.book2D("FlagMap_DIGITIME",
				  "RBX Map of HFDigiTime Flags;RBX;RM",
				  24,131.5,155.5,4,0.5,4.5);
  h_FlagMap_LONGSHORT=ib.book2D("FlagMap_LONGSHORT",
				   "RBX Map of HFLongShort Flags;RBX;RM",
				   24,131.5,155.5,4,0.5,4.5);

  h_FlagMap_TIMEADD=ib.book2D("FlagMap_TIMEADD",
				 "RBX Map of Timing Added Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);
  h_FlagMap_TIMESUBTRACT=ib.book2D("FlagMap_TIMESUBTRACT",
				      "RBX Map of Timing Subtracted Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);
  h_FlagMap_TIMEERROR=ib.book2D("FlagMap_TIMEERROR",
				   "RBX Map of Timing Error Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);

  h_HBflagcounter=ib.book1D("HBflags","HB flags",32,-0.5,31.5);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEHpdHitMultiplicity, "HpdHitMult",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEPulseShape, "PulseShape",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_R1R2, "HSCP R1R2",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_FracLeader, "HSCP FracLeader",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_OuterEnergy, "HSCP OuterEnergy",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_ExpFit, "HSCP ExpFit",1);
  // 2-bit timing counter
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHETimingTrustBits,"TimingTrust1",1);
  h_HBflagcounter->setBinLabel(2+HcalCaloFlagLabels::HBHETimingTrustBits,"TimingTrust2",1);
  //3-bit timing shape cut
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape1",1);
  h_HBflagcounter->setBinLabel(2+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape2",1);
  h_HBflagcounter->setBinLabel(3+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape3",1);

  // common flags
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  h_HBflagcounter->setBinLabel(1+HcalCaloFlagLabels::Fraction2TS,"Frac2TS",1);

  // HE Flag Histograms
  h_HEflagcounter=ib.book1D("HEflags","HE flags",32,-0.5,31.5);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEHpdHitMultiplicity, "HpdHitMult",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHEPulseShape, "PulseShape",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_R1R2, "HSCP R1R2",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_FracLeader, "HSCP FracLeader",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_OuterEnergy, "HSCP OuterEnergy",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HSCP_ExpFit, "HSCP ExpFit",1);
  // 2-bit timing counter
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHETimingTrustBits,"TimingTrust1",1);
  h_HEflagcounter->setBinLabel(2+HcalCaloFlagLabels::HBHETimingTrustBits,"TimingTrust2",1);
  //3-bit timing shape cut
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape1",1);
  h_HEflagcounter->setBinLabel(2+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape2",1);
  h_HEflagcounter->setBinLabel(3+HcalCaloFlagLabels::HBHETimingShapedCutsBits,"TimingShape3",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  h_HEflagcounter->setBinLabel(1+HcalCaloFlagLabels::Fraction2TS,"Frac2TS",1);

  // HO Flag Histograms
  h_HOflagcounter=ib.book1D("HOflags","HO flags",32,-0.5,31.5);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::Fraction2TS,"Frac2TS",1);

  // HF Flag Histograms
  h_HFflagcounter=ib.book1D("HFflags","HF flags",32,-0.5,31.5);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::HFLongShort, "LongShort",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::HFDigiTime, "DigiTime",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::HFTimingTrustBits,"TimingTrust1",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  h_HFflagcounter->setBinLabel(1+HcalCaloFlagLabels::Fraction2TS,"Frac2TS",1);

  h_HBflagcounter->getTH1F()->LabelsOption("v");
  h_HEflagcounter->getTH1F()->LabelsOption("v");
  h_HOflagcounter->getTH1F()->LabelsOption("v");
  h_HFflagcounter->getTH1F()->LabelsOption("v");


  // Diagnostic plots are currently filled for all rechits (no trigger/threshold requirement)
  // hb
  ib.setCurrentFolder(subdir_+"diagnostics/hb");
  
  h_HBTimeVsEnergy=ib.book2D("HBTimeVsEnergy","HB Time Vs Energy (All RecHits);Energy (GeV); time(nS)",100,0,500,40,-100,100);

  h_HBsizeVsLS=ib.bookProfile("HBRecHitsVsLB","HB RecHits vs Luminosity Block",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 100,0,10000);

  h_HBTime=ib.book1D("HB_time","HB RecHit Time",
			int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HBOccupancy=ib.book1D("HB_occupancy",
			     "HB RecHit Occupancy",260,-0.5,2599.5);
  
  //he
  ib.setCurrentFolder(subdir_+"diagnostics/he");

  h_HETimeVsEnergy=ib.book2D("HETimeVsEnergy","HE Time Vs Energy (All RecHits);Energy (GeV); time(nS)",100,0,500,40,-100,100);

  h_HEsizeVsLS=ib.bookProfile("HERecHitsVsLB","HE RecHits vs Luminosity Block",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 100,0,10000);
	
  h_HETime=ib.book1D("HE_time","HE RecHit Time",
			int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HEOccupancy=ib.book1D("HE_occupancy","HE RecHit Occupancy",260,-0.5,2599.5);
   
  // ho
  ib.setCurrentFolder(subdir_+"diagnostics/ho");	

  h_HOTimeVsEnergy=ib.book2D("HOTimeVsEnergy","HO Time Vs Energy (All RecHits);Energy (GeV); time(nS)",100,0,500,40,-100,100);

  h_HOsizeVsLS=ib.bookProfile("HORecHitsVsLB","HO RecHits vs Luminosity Block",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 100,0,10000);
  h_HOTime=ib.book1D("HO_time",
			"HO RecHit Time",
			int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HOOccupancy=ib.book1D("HO_occupancy",
			     "HO RecHit Occupancy",217,-0.5,2169.5);
  
  // hf
  ib.setCurrentFolder(subdir_+"diagnostics/hf");

  h_HFTimeVsEnergy=ib.book2D("HFTimeVsEnergy","HF Time Vs Energy (All RecHits);Energy (GeV); time(nS)",100,0,500,40,-100,100);

  h_HFsizeVsLS=ib.bookProfile("HFRecHitsVsLB",
				 "HF RecHits vs Luminosity Block",
				 NLumiBlocks_,0.5,NLumiBlocks_+0.5,
				 100, 0,10000);	
  h_HFTime=ib.book1D("HF_time","HF RecHit Time",
			int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HFOccupancy=ib.book1D("HF_occupancy","HF RecHit Occupancy",173,-0.5,1729.5);

  return;
} //void HcalRecHitMonitor::setup(...)

void HcalRecHitMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{

  if (debug_>0) std::cout <<"HcalRecHitMonitor::bookHistograms():  task =  '"<<subdir_<<"'"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run, c);
  if (tevt_==0) // create histograms, if they haven't been created already
    this->setup(ib);
  // Clear histograms at the start of each run if not merging runs
  if (mergeRuns_==false)
    this->reset();

  if (tevt_!=0) return;
  // create histograms displaying trigger parameters?  Specify names?
  ib.setCurrentFolder(subdir_+"rechit_parameters");
  std::string tnames="";
  if (HcalHLTBits_.size()>0)
    tnames=HcalHLTBits_[0];
  for (unsigned int i=1;i<HcalHLTBits_.size();++i)
    tnames=tnames + " OR " + HcalHLTBits_[i];
  ib.bookString("HcalHLTriggerRequirements",tnames);
  tnames="";
  if (MinBiasHLTBits_.size()>0)
    tnames=MinBiasHLTBits_[0];
  for (unsigned int i=1;i<MinBiasHLTBits_.size();++i)
    tnames=tnames + " OR " + MinBiasHLTBits_[i];
  ib.bookString("MinBiasHLTriggerRequirements",tnames);
  return;
  
} //void HcalRecHitMonitor::bookHistograms(...)


void HcalRecHitMonitor::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>0) std::cout <<"HcalRecHitMonitor::endRun():  task =  '"<<subdir_<<"'"<<std::endl;

  //Any special fill calls needed?  Shouldn't be necessary; last endLuminosityBlock should do necessary fills
} // void HcalRecHitMonitor::endrun(...)


/* --------------------------- */

void HcalRecHitMonitor::reset()
{
   // REIMPLEMENT FUNCTIONALITY WITHOUT USING GETTER
  /*std::vector<MonitorElement*> hists = dbe_->getAllContents(subdir_);
  for (unsigned int i=0;i<hists.size();++i)
    {
      if (hists[i]->kind()==MonitorElement::DQM_KIND_TH1F ||
	  hists[i]->kind()==MonitorElement::DQM_KIND_TH2F ||
	  hists[i]->kind()==MonitorElement::DQM_KIND_TPROFILE)
	hists[i]->Reset();
    }*/

   h_rechitieta->Reset();
   h_rechitiphi->Reset();

   h_rechitieta_05->Reset();
   h_rechitieta_10->Reset();
   h_rechitieta_25->Reset();
   h_rechitieta_100->Reset();
   h_rechitiphi_05->Reset();
   h_rechitiphi_10->Reset();
   h_rechitiphi_25->Reset();
   h_rechitiphi_100->Reset();

   h_rechitieta_thresh->Reset();
   h_rechitiphi_thresh->Reset();

   h_HBsizeVsLS->Reset();
   h_HEsizeVsLS->Reset();
   h_HOsizeVsLS->Reset();
   h_HFsizeVsLS->Reset();

   h_HBTime->Reset();
   h_HBThreshTime->Reset();
   h_HBOccupancy->Reset();
   h_HBThreshOccupancy->Reset();

   h_HETime->Reset();
   h_HEThreshTime->Reset();
   h_HEOccupancy->Reset();
   h_HEThreshOccupancy->Reset();

   h_HOTime->Reset();
   h_HOThreshTime->Reset();
   h_HOOccupancy->Reset();
   h_HOThreshOccupancy->Reset();

   h_HFTime->Reset();
   h_HFThreshTime->Reset();
   h_HFOccupancy->Reset();
   h_HFThreshOccupancy->Reset();

   h_HBflagcounter->Reset();
   h_HEflagcounter->Reset();
   h_HOflagcounter->Reset();
   h_HFflagcounter->Reset();
  
   h_FlagMap_HPDMULT->Reset();
   h_FlagMap_PULSESHAPE->Reset();
   h_FlagMap_DIGITIME->Reset();
   h_FlagMap_LONGSHORT->Reset();
   h_FlagMap_TIMEADD->Reset();
   h_FlagMap_TIMESUBTRACT->Reset();
   h_FlagMap_TIMEERROR->Reset();
                 
   h_HFLongShort_vs_LS->Reset();
   h_HFDigiTime_vs_LS->Reset();
   h_HBHEHPDMult_vs_LS->Reset();
   h_HBHEPulseShape_vs_LS->Reset();

   h_HF_FlagCorr->Reset();
   h_HBHE_FlagCorr->Reset();

   h_HFtimedifference->Reset();
   h_HFenergydifference->Reset();
   h_HEtimedifference->Reset();
   h_HEenergydifference->Reset();

   h_HF_HcalHLT_weightedtimedifference->Reset();
   h_HF_HcalHLT_energydifference->Reset();
   h_HE_HcalHLT_weightedtimedifference->Reset();
   h_HE_HcalHLT_energydifference->Reset();

   h_LumiPlot_LS_allevents->Reset();
   h_LumiPlot_LS_MinBiasEvents->Reset();
   h_LumiPlot_LS_MinBiasEvents_notimecut->Reset();
   h_LumiPlot_LS_HcalHLTEvents->Reset();
   h_LumiPlot_LS_HcalHLTEvents_notimecut->Reset();

   h_LumiPlot_SumHT_HFPlus_vs_HFMinus->Reset();
   h_LumiPlot_timeHFPlus_vs_timeHFMinus->Reset();

   h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus->Reset();
  
   h_LumiPlot_BX_allevents->Reset();
   h_LumiPlot_BX_MinBiasEvents->Reset();
   h_LumiPlot_BX_MinBiasEvents_notimecut->Reset();
   h_LumiPlot_BX_HcalHLTEvents->Reset();
   h_LumiPlot_BX_HcalHLTEvents_notimecut->Reset();

   h_LumiPlot_MinTime_vs_MinHT->Reset();
   h_LumiPlot_timeHT_HFM->Reset();
   h_LumiPlot_timeHT_HFP->Reset();

   h_TriggeredEvents->Reset();
   h_HFP_weightedTime->Reset();
   h_HFM_weightedTime->Reset();
   h_HEP_weightedTime->Reset();
   h_HEM_weightedTime->Reset();
   h_HBP_weightedTime->Reset();
   h_HBM_weightedTime->Reset();

   h_HBTimeVsEnergy->Reset();
   h_HETimeVsEnergy->Reset();
   h_HOTimeVsEnergy->Reset();
   h_HFTimeVsEnergy->Reset();
   HFP_HFM_Energy->Reset();


}  

void HcalRecHitMonitor::endJob()
{
  if (!enableCleanup_) return;
  HcalBaseDQMonitor::cleanup();
  this->cleanup();
}


/* --------------------------------- */

/*void HcalRecHitMonitor::cleanup()
{
  //Add code to clean out subdirectories
  if (!enableCleanup_) return;
  if (dbe_)
    {
      dbe_->setCurrentFolder(subdir_); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"rechit_parameters"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"rechit_parameters/thresholds"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_AllRecHits"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_AllRecHits/sumplots"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_PassedMinBias"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_PassedMinBias/sumplots"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_PassedHcalHLTriggers"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"Distributions_PassedHcalHLTriggers/passedTechTriggers/"); dbe_->removeContents();

      dbe_->setCurrentFolder(subdir_+"AnomalousCellFlags"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"diagnostics/hb"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"diagnostics/he"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"diagnostics/ho"); dbe_->removeContents();
      dbe_->setCurrentFolder(subdir_+"diagnostics/hf"); dbe_->removeContents();
    }
  return;
} */ // void HcalRecHitMonitor::cleanup()

/* -------------------------------- */

void HcalRecHitMonitor::analyze(edm::Event const&e, edm::EventSetup const&s)
{
  getLogicalMap(s);
  if (debug_>0)  std::cout <<"HcalRecHitMonitor::analyze; debug = "<<debug_<<std::endl;

  HcalBaseDQMonitor::analyze(e,s);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(e.luminosityBlock())==false) return;

  // Get objects
  edm::Handle<HBHERecHitCollection> hbhe_rechit;
  edm::Handle<HORecHitCollection> ho_rechit;
  edm::Handle<HFRecHitCollection> hf_rechit;

  if (!(e.getByToken(tok_hbhe_,hbhe_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hbheRechitLabel_<<" hbhe_rechit not available";
      return;
    }

  if (!(e.getByToken(tok_hf_,hf_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hfRechitLabel_<<" hf_rechit not available";
      return;
    }
  
  if (!(e.getByToken(tok_ho_,ho_rechit)))
    {
      edm::LogWarning("HcalHotCellMonitor")<< hoRechitLabel_<<" ho_rechit not available";
      return;
    }


  h_LumiPlot_LS_allevents->Fill(currentLS);
  h_LumiPlot_BX_allevents->Fill(e.bunchCrossing());
  processEvent(*hbhe_rechit, *ho_rechit, *hf_rechit, e.bunchCrossing(), e);

//  HcalBaseDQMonitor::analyze(e,s);
} // void HcalRecHitMonitor::analyze()


void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits,
				     int BCN,
				     const edm::Event & iEvent
				     )
{

  
  if (debug_>1) std::cout <<"<HcalRecHitMonitor::processEvent> Processing event..."<<std::endl;


  bool passedHcalHLT=false;
  bool passedMinBiasHLT=false;

  edm::Handle<edm::TriggerResults> hltRes;
  if (!(iEvent.getByToken(tok_trigger_,hltRes)))
    {
      if (debug_>0) edm::LogWarning("HcalRecHitMonitor")<<" Could not get HLT results with tag "<<hltresultsLabel_<<std::endl;
    }
  else
    {
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*hltRes);
      const unsigned int nTrig(triggerNames.size());
      for (unsigned int i=0;i<nTrig;++i)
	{
	  // trigger decision is based on 'OR' of any specified trigger names
	  for (unsigned int k=0;k<HcalHLTBits_.size();++k)
	    {
	      // if (triggerNames.triggerName(i)==HcalHLTBits_[k] && hltRes->accept(i))
	      if (triggerNames.triggerName(i).find(HcalHLTBits_[k])!=std::string::npos && hltRes->accept(i))
		{ 
		  passedHcalHLT=true;
		  break;
		}
	    }
	  // repeat for minbias triggers
	  for (unsigned int k=0;k<MinBiasHLTBits_.size();++k)
	    {
	      // if (triggerNames.triggerName(i)==MinBiasHLTBits_[k] && hltRes->accept(i))		
	      if (triggerNames.triggerName(i).find(MinBiasHLTBits_[k])!=std::string::npos && hltRes->accept(i))
		{ 
		  passedMinBiasHLT=true;
		  break;
		}
	    }
	}
    } //else

  if (debug_>2 && passedHcalHLT)  std::cout <<"\t<HcalRecHitMonitor::processEvent>  Passed Hcal HLT trigger  "<<std::endl;
  if (debug_>2 && passedMinBiasHLT)  std::cout <<"\t<HcalRecHitMonitor::processEvent>  Passed MinBias HLT trigger  "<<std::endl;
  
  h_TriggeredEvents->Fill(0); // all events
  if (passedMinBiasHLT) h_TriggeredEvents->Fill(1); // Minbias;
  if (passedHcalHLT)    h_TriggeredEvents->Fill(2); // hcal HLT
  processEvent_rechit(hbHits, hoHits, hfHits,passedHcalHLT,passedMinBiasHLT,BCN);

  return;
} // void HcalRecHitMonitor::processEvent(...)


/* --------------------------------------- */


void HcalRecHitMonitor::processEvent_rechit( const HBHERecHitCollection& hbheHits,
					     const HORecHitCollection& hoHits,
					     const HFRecHitCollection& hfHits,
					     bool passedHcalHLT,
					     bool passedMinBiasHLT,
					     int BCN)
{
  // Gather rechit info
  
  //const float area[]={0.111,0.175,0.175,0.175,0.175,0.175,0.174,0.178,0.172,0.175,0.178,0.346,0.604};

  if (debug_>1) std::cout <<"<HcalRecHitMonitor::processEvent_rechitenergy> Processing rechits..."<<std::endl;
  
  // loop over HBHE
  
  int     hbocc=0;
  int     heocc=0;
  int     hboccthresh=0;
  int     heoccthresh=0;

  double HtPlus =0, HtMinus=0;
  double HFePlus=0, HFeMinus=0;
  double HBePlus=0, HBeMinus=0;
  double HEePlus=0, HEeMinus=0;
  double HFtPlus=0, HFtMinus=0;
  double HBtPlus=0, HBtMinus=0;
  double HEtPlus=0, HEtMinus=0;

  int hbpocc=0, hbmocc=0, hepocc=0, hemocc=0, hfpocc=0, hfmocc=0;

  for (unsigned int i=0;i<4;++i)
    {
      OccupancyByDepth.depth[i]->update();
      OccupancyThreshByDepth.depth[i]->update();
      SumEnergyByDepth.depth[i]->update();
      SqrtSumEnergy2ByDepth.depth[i]->update();
      SumTimeByDepth.depth[i]->update();
    }
    
  h_HBflagcounter->update();
  h_HEflagcounter->update();
  h_HFflagcounter->update();
  h_HOflagcounter->update();
  
  
  for (HBHERecHitCollection::const_iterator HBHEiter=hbheHits.begin(); HBHEiter!=hbheHits.end(); ++HBHEiter) 
    { // loop over all hits
      float en = HBHEiter->energy();
      float ti = HBHEiter->time();
      HcalDetId id(HBHEiter->detid().rawId());
      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();

      if (en>0.5)
        {
          h_rechitieta_05->Fill(ieta);
          h_rechitiphi_05->Fill(iphi);
          if (en>1.)
            {
              h_rechitieta_10->Fill(ieta);
              h_rechitiphi_10->Fill(iphi);
              if (en>2.5)
                {
                  h_rechitieta_25->Fill(ieta);
                  h_rechitiphi_25->Fill(iphi);
                  if (en>10.)
                    {
                      h_rechitieta_100->Fill(ieta);
                      h_rechitiphi_100->Fill(iphi);
                    }
                }
            }
        }



      HcalSubdetector subdet = id.subdet();
      double fEta=fabs(0.5*(theHBHEEtaBounds[abs(ieta)-1]+theHBHEEtaBounds[abs(ieta)]));

      int calcEta = CalcEtaBin(subdet,ieta,depth);
      int rbxindex=logicalMap_->getHcalFrontEndId(HBHEiter->detid()).rbxIndex();
      int rm= logicalMap_->getHcalFrontEndId(HBHEiter->detid()).rm();

      // Fill HBHE flag plots
      h_HBHE_FlagCorr->Fill(HBHEiter->flagField(HcalCaloFlagLabels::HBHEPulseShape),
			    HBHEiter->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity)); 

      if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity))
	{
	  h_FlagMap_HPDMULT->Fill(rbxindex,rm);
	  h_HBHEHPDMult_vs_LS->Fill(currentLS,1);
	}
      if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEPulseShape))
	{
	  h_FlagMap_PULSESHAPE->Fill(rbxindex,rm);
	  h_HBHEPulseShape_vs_LS->Fill(currentLS,1);
	}
      if (HBHEiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
      else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	h_FlagMap_TIMEADD->Fill(rbxindex,rm);
      else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	h_FlagMap_TIMEERROR->Fill(rbxindex,rm);

      if (subdet==HcalBarrel)
	{
	  if (en>HBenergyThreshold_)
	    h_HBTimeVsEnergy->Fill(en,ti);
	  //Looping over HB searching for flags --- cris
	  for (int f=0;f<32;f++)
	    {
	      // Let's display HSCP just to see if these bits are set
	      /*
	       if (f == HcalCaloFlagLabels::HSCP_R1R2)         continue;
               if (f == HcalCaloFlagLabels::HSCP_FracLeader)   continue;
               if (f == HcalCaloFlagLabels::HSCP_OuterEnergy)  continue;
               if (f == HcalCaloFlagLabels::HSCP_ExpFit)       continue;
	      */
	      if (HBHEiter->flagField(f))
		++HBflagcounter_[f];
	    }
	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;
	  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
	    h_HBTime->Fill(ti);
	  else
	    ++HBtime_[int(ti-RECHITMON_TIME_MIN)];
	  ++hbocc; 

	  // Threshold plots;  require E> threshold and minbias trigger
	  if (
	      en>=HBenergyThreshold_ && 
	      en/cosh(fEta)>=HBETThreshold_ 
	      ) 
	    {
	      if (passedMinBiasHLT==true)
		{
		  ++occupancy_thresh_[calcEta][iphi-1][depth-1];
		  energy_thresh_[calcEta][iphi-1][depth-1]+=en;
		  energy2_thresh_[calcEta][iphi-1][depth-1]+=pow(en,2);
		  time_thresh_[calcEta][iphi-1][depth-1]+=ti;
		  
		  ++hboccthresh;
		  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
		    h_HBThreshTime->Fill(ti);
		  else
		    ++HBtime_thresh_[int(ti-RECHITMON_TIME_MIN)];
		}

	      if (ieta>0)
		{
		  HBePlus+=en;
		  HBtPlus+=ti*en;
		  hbpocc++;
		}
	      else
		{
		  HBeMinus+=en;
		  HBtMinus+=ti*en;
		  hbmocc++;
		}
	    } // if (HB  en>thresh, ET>thresh)
	} // if (id.subdet()==HcalBarrel)

      else if (subdet==HcalEndcap)
	{
	  if (en>HEenergyThreshold_)
	    h_HETimeVsEnergy->Fill(en,ti);
	  //Looping over HE searching for flags --- cris
	  for (int f=0;f<32;f++)
            {
              if (HBHEiter->flagField(f))
                ++HEflagcounter_[f];
            }

	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;

	  ++heocc;
	  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
	    h_HETime->Fill(ti);
	  else
	    ++HEtime_[int(ti-RECHITMON_TIME_MIN)];

	  // Threshold plots require e>E_thresh, ET>ET_thresh
	  if (en>=HEenergyThreshold_
	      && en/cosh(fEta)>=HEETThreshold_
	      )
	    {
	      // occupancy plots also require passedMinBiasHLT
	      if (passedMinBiasHLT==true)
		{
		  ++occupancy_thresh_[calcEta][iphi-1][depth-1];
		  energy_thresh_[calcEta][iphi-1][depth-1]+=en;
		  energy2_thresh_[calcEta][iphi-1][depth-1]+=pow(en,2);
		  time_thresh_[calcEta][iphi-1][depth-1]+=ti;
		  ++heoccthresh;
		  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
		      h_HEThreshTime->Fill(ti);
		  else
		    ++HEtime_thresh_[int(ti-RECHITMON_TIME_MIN)];
		}
	      // ePlus, tPlus calculated regardless of trigger
	      if (ieta>0)
		{
		  HEePlus+=en;
		  HEtPlus+=ti*en;
		  hepocc++;
		}
	      else
		{
		  HEeMinus+=en;
		  HEtMinus+=ti*en;
		  hemocc++;
		}
	    } // if (en>=HEenergyThreshold_ && ET>threshold)

	} // else if (id.subdet()==HcalEndcap)
     
    } //for (HBHERecHitCollection::const_iterator HBHEiter=...)
  
  // Calculate normalized time
  HEePlus>0  ?  HEtPlus/=HEePlus   :  HEtPlus=10000;
  HEeMinus>0 ?  HEtMinus/=HEeMinus :  HEtMinus=-10000;
  HBePlus>0  ?  HBtPlus/=HBePlus   :  HBtPlus=10000;
  HBeMinus>0 ?  HBtMinus/=HBeMinus :  HBtMinus=-10000;
  
  ++HB_occupancy_[hbocc/10];
  ++HE_occupancy_[heocc/10];
  ++HB_occupancy_thresh_[hboccthresh/10];
  ++HE_occupancy_thresh_[heoccthresh/10];
  h_HBsizeVsLS->Fill(currentLS,hbocc);
  h_HEsizeVsLS->Fill(currentLS,heocc);

  // loop over HO

  h_HOsizeVsLS->Fill(currentLS,hoHits.size());
  int hoocc=0;
  int hooccthresh=0;
  for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
    { // loop over all hits
      float en = HOiter->energy();
      float ti = HOiter->time();
      if (en>HOenergyThreshold_)
	h_HOTimeVsEnergy->Fill(en,ti);

      HcalDetId id(HOiter->detid().rawId());
      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();

      if (en>0.5)
        {
          h_rechitieta_05->Fill(ieta);
          h_rechitiphi_05->Fill(iphi);
          if (en>1.)
            {
              h_rechitieta_10->Fill(ieta);
              h_rechitiphi_10->Fill(iphi);
              if (en>2.5)
                {
                  h_rechitieta_25->Fill(ieta);
                  h_rechitiphi_25->Fill(iphi);
                  if (en>10.)
                    {
                      h_rechitieta_100->Fill(ieta);
                      h_rechitiphi_100->Fill(iphi);
                    }
                }
            }
        }



      int calcEta = CalcEtaBin(HcalOuter,ieta,depth);
      double fEta=fabs(0.5*(theHBHEEtaBounds[abs(ieta)-1]+theHBHEEtaBounds[abs(ieta)]));
      
      int rbxindex=logicalMap_->getHcalFrontEndId(HOiter->detid()).rbxIndex();
      int rm= logicalMap_->getHcalFrontEndId(HOiter->detid()).rm();
      
      if (HOiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
      else if (HOiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	h_FlagMap_TIMEADD->Fill(rbxindex,rm);
      else if (HOiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	h_FlagMap_TIMEERROR->Fill(rbxindex,rm);
      
      
      //Looping over HO searching for flags --- cris
      for (int f=0;f<32;f++)
	{
	  if (HOiter->flagField(f))
	    HOflagcounter_[f]++;
	}
      
      ++occupancy_[calcEta][iphi-1][depth-1];
      energy_[calcEta][iphi-1][depth-1]+=en;
      energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
      time_[calcEta][iphi-1][depth-1]+=ti;
      ++hoocc;
      if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
	h_HOTime->Fill(ti);
      else
	++HOtime_[int(ti-RECHITMON_TIME_MIN)];

      // We don't calculate HOplus/HOminus values (independent of trigger), so require min bias trigger 
      // along with E, ET thresholds directly in this HO loop:

      if (en>=HOenergyThreshold_  
	  && en/cosh(fEta)>=HOETThreshold_
	  && passedMinBiasHLT==true   
	  )
	{
	  ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	  energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	  energy2_thresh_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_thresh_[calcEta][iphi-1][depth-1]+=ti;

	  ++hooccthresh;
	  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
	    h_HOThreshTime->Fill(ti);
	  else
	    ++HOtime_thresh_[int(ti-RECHITMON_TIME_MIN)];
	} 
    } // loop over all HO hits

  ++HO_occupancy_[hoocc/10];
  ++HO_occupancy_thresh_[hooccthresh/10];
   
  // loop over HF
  h_HFsizeVsLS->Fill(currentLS,hfHits.size());

  HtPlus=0; HtMinus=0;
  
  int hfocc=0;
  int hfoccthresh=0;
  for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
    { // loop over all hits
      float en = HFiter->energy();
      float ti = HFiter->time();
      if (en> HFenergyThreshold_)
	h_HFTimeVsEnergy->Fill(en,ti);

      HcalDetId id(HFiter->detid().rawId());
      int ieta = id.ieta();
      int iphi = id.iphi();
      int depth = id.depth();

      if (en>0.5)
	{
	  h_rechitieta_05->Fill(ieta);
	  h_rechitiphi_05->Fill(iphi);
	  if (en>1.)
	    {
	      h_rechitieta_10->Fill(ieta);
	      h_rechitiphi_10->Fill(iphi);
	      if (en>2.5)
		{
		  h_rechitieta_25->Fill(ieta);
		  h_rechitiphi_25->Fill(iphi);
		  if (en>10.)
		    {
		      h_rechitieta_100->Fill(ieta);
		      h_rechitiphi_100->Fill(iphi);
		    }
		}
	    }
	}

      double fEta=fabs(0.5*(theHFEtaBounds[abs(ieta)-29]+theHFEtaBounds[abs(ieta)-28]));
      int calcEta = CalcEtaBin(HcalForward,ieta,depth);

      int rbxindex=logicalMap_->getHcalFrontEndId(HFiter->detid()).rbxIndex();
      int rm= logicalMap_->getHcalFrontEndId(HFiter->detid()).rm(); 
	 
      h_HF_FlagCorr->Fill(HFiter->flagField(HcalCaloFlagLabels::HFDigiTime),HFiter->flagField(HcalCaloFlagLabels::HFLongShort)); 
      if (HFiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
      else if (HFiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	h_FlagMap_TIMEADD->Fill(rbxindex,rm);
      else if (HFiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	h_FlagMap_TIMEERROR->Fill(rbxindex,rm);

      if (HFiter->flagField(HcalCaloFlagLabels::HFDigiTime))
	{
	  h_FlagMap_DIGITIME->Fill(rbxindex,rm);
	  h_HFDigiTime_vs_LS->Fill(currentLS,1);
	}
      if (HFiter->flagField(HcalCaloFlagLabels::HFLongShort))
	{
	  h_FlagMap_LONGSHORT->Fill(rbxindex,rm);
	  h_HFLongShort_vs_LS->Fill(currentLS,1);
	}
      //Looping over HF searching for flags --- cris
      for (int f=0;f<32;f++)
	{
	  if (HFiter->flagField(f))
	    HFflagcounter_[f]++;
	}

      // Occupancy plots, without threshold
      ++occupancy_[calcEta][iphi-1][depth-1];
      energy_[calcEta][iphi-1][depth-1]+=en;
      energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
      time_[calcEta][iphi-1][depth-1]+=ti;
      ++hfocc;
      if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
	h_HFTime->Fill(ti);
      else
	++HFtime_[int(ti-RECHITMON_TIME_MIN)];

      ieta>0 ?  HtPlus+=en/cosh(fEta)  : HtMinus+=en/cosh(fEta);  // add energy from all cells, or only those > threshold?

      if (en>=HFenergyThreshold_ && 
	  en/cosh(fEta)>=HFETThreshold_ 
	  )
	{
	  // Occupancy plots require min bias trigger, along with thresholds exceeded
	  if (passedMinBiasHLT)
	    {
	      ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	      energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	      energy2_thresh_[calcEta][iphi-1][depth-1]+=pow(en,2);
	      time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	      
	      ++hfoccthresh;
	      if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
		h_HFThreshTime->Fill(ti);
	      else
		++HFtime_thresh_[int(ti-RECHITMON_TIME_MIN)];
	    }

	  if (ieta>0)
	    {
	      HFtPlus+=en*ti;
	      HFePlus+=en;
	      hfpocc++;
	    }
	  else if (ieta<0)
	    {
	      HFtMinus+=en*ti;
	      HFeMinus+=en;
	      hfmocc++;
	    }
	} // if (en>thresh, ET>thresh)
    } // loop over all HF hits
     
  ++HF_occupancy_[hfocc/10];
  ++HF_occupancy_thresh_[hfoccthresh/10];

  
  // Form event-wide variables (time averages, etc.), and plot them

  // Calculate weighted times.  (Set tPlus, tMinus to overflow in case where total energy < 0)
  HFePlus>0  ? HFtPlus/=HFePlus    : HFtPlus  =  10000;
  HFeMinus>0 ? HFtMinus/=HFeMinus  : HFtMinus = -10000;
     
  double mintime=99;  // used to be min(tPlus,tMinus);
  double minHT=std::min(HtMinus,HtPlus);
  minHT==HtMinus ?  mintime=HFtMinus : mintime = HFtPlus;
  //mintime = min(HFtPlus,HFtMinus); // I think we might want to use this value for mintime?


  h_LumiPlot_MinTime_vs_MinHT->Fill(minHT, mintime);
  h_LumiPlot_timeHT_HFM->Fill(HtMinus,HFtMinus);
  h_LumiPlot_timeHT_HFP->Fill(HtPlus,HFtPlus);

  if (passedMinBiasHLT==true)
    {
      h_LumiPlot_SumHT_HFPlus_vs_HFMinus->Fill(HtMinus,HtPlus);
      // HtMinus, HtPlus require no energy cuts for their contributing cells
      // HFeMinus, HFePlus require that cells be > threshold cut
      
      if (HtMinus>1 && HtPlus > 1) // is this the condition we want, or do we want hfmocc>0 && hfpocc >0?
      	{
	  h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus->Fill(HFeMinus,HFePlus);
	  h_LumiPlot_timeHFPlus_vs_timeHFMinus->Fill(HFtMinus,HFtPlus);

	  h_HFP_weightedTime->Fill(HFtPlus);
	  h_HFM_weightedTime->Fill(HFtMinus);
	  h_HBP_weightedTime->Fill(HBtPlus);
	  h_HBM_weightedTime->Fill(HBtMinus);
	  
	  h_HEP_weightedTime->Fill(HEtPlus);
	  h_HEM_weightedTime->Fill(HEtMinus);

	  if (hepocc>0 && hemocc>0)
	    {
	      h_HEtimedifference->Fill(HEtPlus-HEtMinus);
	      if (HEePlus-HEeMinus!=0) h_HEenergydifference->Fill((HEePlus-HEeMinus)/(HEePlus+HEeMinus));
	    }
	  if (hfpocc>0 && hfmocc>0)  // which condition do we want?
	    {
	      h_HFtimedifference->Fill((HFtPlus)-(HFtMinus));
	      if (HFePlus+HFeMinus!=0) h_HFenergydifference->Fill((HFePlus-HFeMinus)/(HFePlus+HFeMinus));
	    }

	  h_LumiPlot_LS_MinBiasEvents_notimecut->Fill(currentLS);
	  h_LumiPlot_BX_MinBiasEvents_notimecut->Fill(BCN);
	  if (fabs(HFtPlus-HFtMinus)<timediffThresh_)
	    {
	      h_LumiPlot_LS_MinBiasEvents->Fill(currentLS);
	      h_LumiPlot_BX_MinBiasEvents->Fill(BCN);
	    }
	  
	  HFP_HFM_Energy->Fill(HFeMinus/1000., HFePlus/1000.);
	}
 
      if (debug_>1) std::cout <<"\t<HcalRecHitMonitor:: HF averages>  TPLUS = "<<HFtPlus<<"  EPLUS = "<<HFePlus<<"  TMINUS = "<<HFtMinus<<"  EMINUS = "<<HFeMinus<<"  Weighted Time Diff = "<<((HFtPlus)-(HFtMinus))<<std::endl;
      

    } // if (passedMinBiasHLT)

  if (passedHcalHLT && HtMinus>1 && HtPlus> 1 )
    {
      if (hfpocc>0 && hfmocc>0)
	{
	  h_HF_HcalHLT_weightedtimedifference->Fill(HFtPlus-HFtMinus);
	  if (HFePlus+HFeMinus!=0) h_HF_HcalHLT_energydifference->Fill((HFePlus-HFeMinus)/(HFePlus+HFeMinus));
	}
      if  (hepocc>0 && hemocc>0)
	{
	  h_HE_HcalHLT_weightedtimedifference->Fill(HEtPlus-HEtMinus);
	  if (HEePlus-HEeMinus!=0) h_HE_HcalHLT_energydifference->Fill((HEePlus-HEeMinus)/(HEePlus+HEeMinus));
	}

      h_LumiPlot_LS_HcalHLTEvents_notimecut->Fill(currentLS);
      h_LumiPlot_BX_HcalHLTEvents_notimecut->Fill(BCN);
      if (fabs(HFtPlus-HFtMinus)<timediffThresh_)
	{
	  h_LumiPlot_LS_HcalHLTEvents->Fill(currentLS);
	  h_LumiPlot_BX_HcalHLTEvents->Fill(BCN);
	}
     } // passsed Hcal HLT
   
 return;
} // void HcalRecHitMonitor::processEvent_rechitenergy

/* --------------------------------------- */


void HcalRecHitMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                                           const edm::EventSetup& c)

{
  // don't fill lumi block information if it's already been filled
  if (LumiInOrder(lumiSeg.luminosityBlock())==false) return;
  fill_Nevents();
  return;
} //endLuminosityBlock


void HcalRecHitMonitor::fill_Nevents(void)
{
  // looking at the contents of HbFlagcounters
  if (debug_>0)
    {
      for (int k = 0; k < 32; k++){
	std::cout << "<HcalRecHitMonitor::fill_Nevents>  HF Flag counter:  Bin #" << k+1 << " = "<< HFflagcounter_[k] << std::endl;
      }
    }

  for (int i=0;i<32;i++)
    {
      h_HBflagcounter->Fill(i,HBflagcounter_[i]);
      h_HEflagcounter->Fill(i,HEflagcounter_[i]);
      h_HOflagcounter->Fill(i,HOflagcounter_[i]);
      h_HFflagcounter->Fill(i,HFflagcounter_[i]);
      HBflagcounter_[i]=0;
      HEflagcounter_[i]=0;
      HOflagcounter_[i]=0;
      HFflagcounter_[i]=0;
    }

  // Fill Occupancy & Sum Energy, Time plots
  int myieta=-1;
  if (ievt_>0)
    {
      for (int mydepth=0;mydepth<4;++mydepth)
	{
	  for (int eta=0;eta<OccupancyByDepth.depth[mydepth]->getNbinsX();++eta)
	    {
	      myieta=CalcIeta(eta,mydepth+1);

	      for (int phi=0;phi<72;++phi)
		{
		  if (occupancy_[eta][phi][mydepth]>0)
		    {
		      h_rechitieta->Fill(myieta,occupancy_[eta][phi][mydepth]);
		      h_rechitiphi->Fill(phi+1,occupancy_[eta][phi][mydepth]);
		    }
		  if (occupancy_thresh_[eta][phi][mydepth]>0)
		    {
		      h_rechitieta_thresh->Fill(myieta,occupancy_thresh_[eta][phi][mydepth]);
		      h_rechitiphi_thresh->Fill(phi+1,occupancy_thresh_[eta][phi][mydepth]);
		    }
		  OccupancyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_[eta][phi][mydepth]);
		  SumEnergyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_[eta][phi][mydepth]);
                  SqrtSumEnergy2ByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,sqrt(energy2_[eta][phi][mydepth]));
		  SumTimeByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_[eta][phi][mydepth]);

		  OccupancyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_thresh_[eta][phi][mydepth]);
		  SumEnergyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_thresh_[eta][phi][mydepth]);
		  SqrtSumEnergy2ThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,sqrt(energy2_thresh_[eta][phi][mydepth]));
		  SumTimeThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_thresh_[eta][phi][mydepth]);
		} // for (int phi=0;phi<72;++phi)
	    } // for (int eta=0;eta<OccupancyByDepth...;++eta)
	} // for (int mydepth=0;...)

      FillUnphysicalHEHFBins(OccupancyByDepth);
      FillUnphysicalHEHFBins(OccupancyThreshByDepth);
      FillUnphysicalHEHFBins(SumEnergyByDepth);
      FillUnphysicalHEHFBins(SqrtSumEnergy2ByDepth);
      FillUnphysicalHEHFBins(SumEnergyThreshByDepth);
      FillUnphysicalHEHFBins(SqrtSumEnergy2ThreshByDepth);
      FillUnphysicalHEHFBins(SumTimeByDepth);
      FillUnphysicalHEHFBins(SumTimeThreshByDepth);

    } // if (ievt_>0)

  // Fill subdet plots

  for (int i=0;i<(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN);++i)
    {
      if (HBtime_[i]!=0)
	{
	  h_HBTime->setBinContent(i+1,HBtime_[i]);
	}
      if (HBtime_thresh_[i]!=0)
	{
	  h_HBThreshTime->setBinContent(i+1,HBtime_thresh_[i]);
	}
      if (HEtime_[i]!=0)
	{

	  h_HETime->setBinContent(i+1,HEtime_[i]);
	}
      if (HEtime_thresh_[i]!=0)
	{
	  h_HEThreshTime->setBinContent(i+1,HEtime_thresh_[i]);
	}
      if (HOtime_[i]!=0)
	{
	  h_HOTime->setBinContent(i+1,HOtime_[i]);
	}
      if (HOtime_thresh_[i]!=0)
	{
	  h_HOThreshTime->setBinContent(i+1,HOtime_thresh_[i]);
	}
      if (HFtime_[i]!=0)
	{
	  h_HFTime->setBinContent(i+1,HFtime_[i]);
	}
      if (HFtime_thresh_[i]!=0)
	{
	  h_HFThreshTime->setBinContent(i+1,HFtime_thresh_[i]);
	}
    } // for (int  i=0;i<(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN);++i)

  for (int i=0;i<260;++i)
    {
      if (HB_occupancy_[i]>0)
	{
	  h_HBOccupancy->setBinContent(i+1,HB_occupancy_[i]);
	}
      if (HB_occupancy_thresh_[i]>0)
	{
	  h_HBThreshOccupancy->setBinContent(i+1,HB_occupancy_thresh_[i]);
	}
      if (HE_occupancy_[i]>0)
	{
	  h_HEOccupancy->setBinContent(i+1,HE_occupancy_[i]);
	}
      if (HE_occupancy_thresh_[i]>0)
	{
	  h_HEThreshOccupancy->setBinContent(i+1,HE_occupancy_thresh_[i]);
	}
    }//for (int i=0;i<260;++i)

  for (int i=0;i<217;++i)
    {
      if (HO_occupancy_[i]>0)
	{
	  h_HOOccupancy->setBinContent(i+1,HO_occupancy_[i]);
	}
      if (HO_occupancy_thresh_[i]>0)
	{
	  h_HOThreshOccupancy->setBinContent(i+1,HO_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<217;++i)

  for (int i=0;i<173;++i)
    {
      if (HF_occupancy_[i]>0)
	{
	  h_HFOccupancy->setBinContent(i+1,HF_occupancy_[i]);
	}
      if (HF_occupancy_thresh_[i]>0)
	{
	  h_HFThreshOccupancy->setBinContent(i+1,HF_occupancy_thresh_[i]);
	}
    }//  for (int i=0;i<173;++i)

  //zeroCounters();

  if (debug_>0)
    std::cout <<"<HcalRecHitMonitor::fill_Nevents> FILLED REC HIT CELL PLOTS"<<std::endl;

} // void HcalRecHitMonitor::fill_Nevents(void)


void HcalRecHitMonitor::zeroCounters(void)
{
  // Set all histogram counters back to zero

  for (int i=0;i<32;++i)
    {
      HBflagcounter_[i]=0;
      HEflagcounter_[i]=0;
      HOflagcounter_[i]=0;
      HFflagcounter_[i]=0;

    }
  // TH2F counters
  for (int i=0;i<85;++i)
    {
      for (int j=0;j<72;++j)
	{
	  for (int k=0;k<4;++k)
	    {
	      occupancy_[i][j][k]=0;
	      occupancy_thresh_[i][j][k]=0;
	      energy_[i][j][k]=0;
              energy2_[i][j][k]=0;
	      energy_thresh_[i][j][k]=0;
	      energy2_thresh_[i][j][k]=0;
	      time_[i][j][k]=0;
	      time_thresh_[i][j][k]=0;
	    }
	} // for (int j=0;j<PHIBINS;++j)
    } // for (int i=0;i<87;++i)

  // TH1F counters
  
  for (int i=0;i<200;++i)
    {
      HFenergyLong_[i]=0;
      HFenergyLong_thresh_[i]=0;
      HFenergyShort_[i]=0;
      HFenergyShort_thresh_[i]=0;
    }

  // time
  for (int i=0;i<(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN);++i)
    {
      HBtime_[i]=0;
      HBtime_thresh_[i]=0;
      HEtime_[i]=0;
      HEtime_thresh_[i]=0;
      HOtime_[i]=0;
      HOtime_thresh_[i]=0;
      HFtime_[i]=0;
      HFtime_thresh_[i]=0;
      HFtimeLong_[i]=0;
      HFtimeLong_thresh_[i]=0;
      HFtimeShort_[i]=0;
      HFtimeShort_thresh_[i]=0;
    }

  // occupancy
  for (int i=0;i<865;++i)
    {
      if (i<260)
	{
	  HB_occupancy_[i]=0;
	  HE_occupancy_[i]=0;
	  HB_occupancy_thresh_[i]=0;
	  HE_occupancy_thresh_[i]=0;
	}
      if (i<218)
	{
	  HO_occupancy_[i]=0;
	  HO_occupancy_thresh_[i]=0;
	}
      if (i<174)
	{
	  HF_occupancy_[i]=0;
	  HF_occupancy_thresh_[i]=0;
	}

      HFlong_occupancy_[i] =0;
      HFshort_occupancy_[i]=0;
      HFlong_occupancy_thresh_[i] =0;
      HFshort_occupancy_thresh_[i]=0;
    } // for (int i=0;i<865;++i)

  return;
} //void HcalRecHitMonitor::zeroCounters(void)

DEFINE_FWK_MODULE(HcalRecHitMonitor);
