#include "DQM/HcalMonitorTasks/interface/HcalRecHitMonitor.h"
#include "FWCore/Common/interface/TriggerNames.h"

using namespace std;

HcalRecHitMonitor::HcalRecHitMonitor()
{
} //constructor

HcalRecHitMonitor::~HcalRecHitMonitor()
{
} //destructor


/* ------------------------------------ */ 


void HcalRecHitMonitor::setup(const edm::ParameterSet& ps,
			      DQMStore* dbe)
{
  HcalBaseMonitor::setup(ps,dbe);

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
  if (fVerbosity>0)
    std::cout <<"<HcalRecHitMonitor::setup>  Setting up histograms"<<endl;

  // Can we include this just in the setup, or do we need to get a new logical map with every run?
  HcalLogicalMapGenerator gen;
  logicalMap=new HcalLogicalMap(gen.createMap());

  baseFolder_ = rootFolder_+"RecHitMonitor_Hcal";

  // Assume subdetectors not present until shown otherwise
  HBpresent_=false;
  HEpresent_=false;
  HOpresent_=false;
  HFpresent_=false;

  // RecHit Monitor - specific cfg variables

  if (fVerbosity>1)
    std::cout <<"<HcalRecHitMonitor::setup>  Getting variable values from cfg files"<<endl;
  
  // Set checkNevents values
  rechit_checkNevents_ = ps.getUntrackedParameter<int>("RecHitMonitor_checkNevents",checkNevents_);
  rechit_minErrorFlag_ = ps.getUntrackedParameter<double>("RecHitMonitor_minErrorFlag",0.0);

  energyThreshold_       = ps.getUntrackedParameter<double>("RecHitMonitor_energyThreshold",                  0);
  HBenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HB_energyThreshold",energyThreshold_);
  HEenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HE_energyThreshold",energyThreshold_);
  HOenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HO_energyThreshold",energyThreshold_);
  HFenergyThreshold_     = ps.getUntrackedParameter<double>("RecHitMonitor_HF_energyThreshold",energyThreshold_);

  collisionHFthresh_ = ps.getUntrackedParameter<double>("RecHitMonitor_collisionHFthresh",3);
  collisionHEthresh_ = ps.getUntrackedParameter<double>("RecHitMonitor_collisionHEthresh",3);

  // Set allowed types of events for running through rechitmon
  AllowedCalibTypes_ = ps.getUntrackedParameter<vector<int> >("RecHitMonitor_AllowedCalibTypes",AllowedCalibTypes_);

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor SETUP -> "<<cpu_timer.cpuTime()<<endl;
    }
  return;
} //void HcalRecHitMonitor::setup(...)

void HcalRecHitMonitor::beginRun()
{
  HcalBaseMonitor::beginRun();
  zeroCounters();
  if (!m_dbe) return;
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }

  if (fVerbosity>1)
    std::cout <<"<HcalRecHitMonitor::beginRun>  Setting up histograms"<<endl;

  m_dbe->setCurrentFolder(baseFolder_);
  meEVT_ = m_dbe->bookInt("RecHit Task Event Number");
  meTOTALEVT_ = m_dbe->bookInt("RecHit Task Total Events Processed");

  m_dbe->setCurrentFolder(baseFolder_+"/rechit_info");

  SetupEtaPhiHists(OccupancyByDepth,"RecHit Occupancy","");;
  
  h_HFtimedifference = m_dbe->book1D("HFweightedtimeDifference","Energy-Weighted time difference between HF+ and HF-",251,-250.5,250.5);
  h_HEtimedifference = m_dbe->book1D("HEweightedtimeDifference","Energy-Weighted time difference between HE+ and HE-",251,-250.5,250.5);
  h_HFrawtimedifference = m_dbe->book1D("HFtimeDifference","Average Time difference between HF+ and HF-",251,-250.5,250.5);
  h_HErawtimedifference = m_dbe->book1D("HEtimeDifference","Average Time difference between HE+ and HE-",251,-250.5,250.5);

  h_HFenergydifference = m_dbe->book1D("HFenergyDifference","Sum(E_HFPlus - E_HFMinus)/Sum(E_HFPlus + E_HFMinus)",200,-1,1);
  h_HEenergydifference = m_dbe->book1D("HEenergyDifference","Sum(E_HEPlus - E_HEMinus)/Sum(E_HEPlus + E_HEMinus)",200,-1,1);
  h_HFrawenergydifference = m_dbe->book1D("HFaverageenergyDifference","E_HFPlus - E_HFMinus (energy averaged over rechits)",500,-100,100);
  h_HErawenergydifference = m_dbe->book1D("HEaverageenergyDifference","E_HEPlus - E_HEMinus (energy averaged over rechits)",500,-100,100);
  
  h_HFnotBPTXtimedifference = m_dbe->book1D("HFnotBPTXweightedtimeDifference","Energy-Weighted time difference between HF+ and HF-",251,-250.5,250.5);
  h_HEnotBPTXtimedifference = m_dbe->book1D("HEnotBPTXweightedtimeDifference","Energy-Weighted time difference between HE+ and HE-",251,-250.5,250.5);
  h_HFnotBPTXrawtimedifference = m_dbe->book1D("HFnotBPTXtimeDifference","Average Time difference between HF+ and HF-",251,-250.5,250.5);
  h_HEnotBPTXrawtimedifference = m_dbe->book1D("HEnotBPTXtimeDifference","Average Time difference between HE+ and HE-",251,-250.5,250.5);

  h_HFnotBPTXenergydifference = m_dbe->book1D("HFnotBPTXenergyDifference","Sum(E_HFPlus - E_HFMinus)/Sum(E_HFPlus + E_HFMinus)",200,-1,1);
  h_HEnotBPTXenergydifference = m_dbe->book1D("HEnotBPTXenergyDifference","Sum(E_HEPlus - E_HEMinus)/Sum(E_HEPlus + E_HEMinus)",200,-1,1);
  h_HFnotBPTXrawenergydifference = m_dbe->book1D("HFnotBPTXaverageenergyDifference","E_HFPlus - E_HFMinus (energy averaged over rechits)",500,-100,100);
  h_HEnotBPTXrawenergydifference = m_dbe->book1D("HEnotBPTXaverageenergyDifference","E_HEPlus - E_HEMinus (energy averaged over rechits)",500,-100,100);
  
  m_dbe->setCurrentFolder(baseFolder_+"/luminosityplots");
  h_LumiPlot_EventsPerLS=m_dbe->book1D("EventsPerLS","Number of Events with HF+ and HF- HT>1 GeV vs LS",Nlumiblocks_,0.5,Nlumiblocks_+0.5); 
  h_LumiPlot_EventsPerLS_notimecut=m_dbe->book1D("EventsPerLS_notimecut","Number of Events with HF+ and HF- HT>1 GeV (no time cut) vs LS",Nlumiblocks_,0.5,Nlumiblocks_+0.5); 

  h_LumiPlot_SumHT_HFPlus_vs_HFMinus = m_dbe->book2D("SumHT_plus_minus","Sum HT for HF+ vs HF-",60,0,30,60,0,30);
  h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus = m_dbe->book2D("SumEnergy_plus_minus","Sum Energy for HF+ vs HF-",60,0,150,60,0,150);
  h_LumiPlot_timeHFPlus_vs_timeHFMinus = m_dbe->book2D("timeHFplus_vs_timeHFminus","Energy-weighted time average of HF+ vs HF-",120,-120,120,120,-120,120);

  h_LumiPlot_MinTime_vs_MinHT = m_dbe->book2D("MinTime_vs_MinHT","(HF+,HF-) Min Time vs Min HT",100,0,10,80,-40,40);
  h_LumiPlot_LS_allevents = m_dbe->book1D("LS_allevents","LS # of all events",Nlumiblocks_,0.5,Nlumiblocks_+0.5);

  h_LumiPlot_BX_allevents = m_dbe->book1D("BX_allevents","BX # of all events",3600,0,3600);
  h_LumiPlot_BX_goodevents = m_dbe->book1D("BX_goodevents","BX # of good events",3600,0,3600);
  h_LumiPlot_BX_goodevents_notimecut = m_dbe->book1D("BX_goodevents_notimecut","BX # of good events (no time cut)",3600,0,3600);
							  

  
  m_dbe->setCurrentFolder(baseFolder_+"/rechit_info/sumplots");
  SetupEtaPhiHists(SumEnergyByDepth,"RecHit Summed Energy","GeV");
  SetupEtaPhiHists(SqrtSumEnergy2ByDepth,"RecHit Sqrt Summed Energy2","GeV");
  SetupEtaPhiHists(SumTimeByDepth,"RecHit Summed Time","nS");

  m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold");
  SetupEtaPhiHists(OccupancyThreshByDepth,"Above Threshold RecHit Occupancy","");
  MonitorElement* THR=m_dbe->bookFloat("HB_Rechit_Energy_Threshold");
  THR->Fill(HBenergyThreshold_);
  THR=m_dbe->bookFloat("HE_Rechit_Energy_Threshold");
  THR->Fill(HEenergyThreshold_);
  THR=m_dbe->bookFloat("HO_Rechit_Energy_Threshold");
  THR->Fill(HOenergyThreshold_);
  THR=m_dbe->bookFloat("HF_Rechit_Energy_Threshold");
  THR->Fill(HFenergyThreshold_);

  m_dbe->setCurrentFolder(baseFolder_+"/rechit_info_threshold/sumplots");
  SetupEtaPhiHists(SumEnergyThreshByDepth,"Above Threshold RecHit Summed Energy","GeV");
  SetupEtaPhiHists(SumTimeThreshByDepth,"Above Threshold RecHit Summed Time","nS");
  
  m_dbe->setCurrentFolder(baseFolder_+"/AnomalousCellFlags");// HB Flag Histograms

  h_HF_FlagCorr=m_dbe->book2D("HF_FlagCorrelation","HF LongShort vs. DigiTime flags; DigiTime; LongShort", 2,-0.5,1.5,2,-0.5,1.5);
  h_HF_FlagCorr->setBinLabel(1,"OFF",1);
  h_HF_FlagCorr->setBinLabel(2,"ON",1);
  h_HF_FlagCorr->setBinLabel(1,"OFF",2);
  h_HF_FlagCorr->setBinLabel(2,"ON",2);

  h_HBHE_FlagCorr=m_dbe->book2D("HBHE_FlagCorrelation","HBHE HpdHitMultiplicity vs. PulseShape flags; PulseShape; HpdHitMultiplicity", 2,-0.5,1.5,2,-0.5,1.5);
  h_HBHE_FlagCorr->setBinLabel(1,"OFF",1);
  h_HBHE_FlagCorr->setBinLabel(2,"ON",1);
  h_HBHE_FlagCorr->setBinLabel(1,"OFF",2);
  h_HBHE_FlagCorr->setBinLabel(2,"ON",2);

  h_FlagMap_HPDMULT=m_dbe->book2D("FlagMap_HPDMULT","RBX Map of HBHEHpdHitMultiplicity Flags;RBX;RM",
				  72,-0.5,71.5,4,0.5,4.5);
  h_FlagMap_PULSESHAPE=m_dbe->book2D("FlagMap_PULSESHAPE","RBX Map of HBHEPulseShape Flags;RBX;RM",
				  72,-0.5,71.5,4,0.5,4.5);
  h_FlagMap_DIGITIME=m_dbe->book2D("FlagMap_DIGITIME","RBX Map of HFDigiTime Flags;RBX;RM",
				  24,131.5,155.5,4,0.5,4.5);
  h_FlagMap_LONGSHORT=m_dbe->book2D("FlagMap_LONGSHORT","RBX Map of HFLongShort Flags;RBX;RM",
				   24,131.5,155.5,4,0.5,4.5);

  h_FlagMap_TIMEADD=m_dbe->book2D("FlagMap_TIMEADD","RBX Map of Timing Added Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);
  h_FlagMap_TIMESUBTRACT=m_dbe->book2D("FlagMap_TIMESUBTRACT","RBX Map of Timing Subtracted Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);
  h_FlagMap_TIMEERROR=m_dbe->book2D("FlagMap_TIMEERROR","RBX Map of Timing Error Flags;RBX;RM",
				   156,-0.5,155.5,4,0.5,4.5);

  h_HBflagcounter=m_dbe->book1D("HBflags","HB flags",32,-0.5,31.5);
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
  h_HEflagcounter=m_dbe->book1D("HEflags","HE flags",32,-0.5,31.5);
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
  h_HOflagcounter=m_dbe->book1D("HOflags","HO flags",32,-0.5,31.5);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingSubtractedBit, "Subtracted",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingAddedBit, "Added",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::TimingErrorBit, "TimingError",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::ADCSaturationBit, "Saturation",1);
  h_HOflagcounter->setBinLabel(1+HcalCaloFlagLabels::Fraction2TS,"Frac2TS",1);

  // HF Flag Histograms
  h_HFflagcounter=m_dbe->book1D("HFflags","HF flags",32,-0.5,31.5);
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
  
  // hb
  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hb");

  h_HBsizeVsLS=m_dbe->bookProfile("HBRecHitsVsLB","HB RecHits vs Luminosity Block",
				  Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				  0,10000);

  h_HBTime=m_dbe->book1D("HB_time","HB RecHit Time",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HBThreshTime=m_dbe->book1D("HB_time_thresh", "HB RecHit Time Above Threshold",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HBOccupancy=m_dbe->book1D("HB_occupancy","HB RecHit Occupancy",260,-0.5,2599.5);
  h_HBThreshOccupancy=m_dbe->book1D("HB_occupancy_thresh","HB RecHit Occupancy Above Threshold",260,-0.5,2599.5);
  
  //he
  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/he");

  h_HEsizeVsLS=m_dbe->bookProfile("HERecHitsVsLB","HE RecHits vs Luminosity Block",
				  Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				  0,10000);
	
  h_HETime=m_dbe->book1D("HE_time","HE RecHit Time",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HEThreshTime=m_dbe->book1D("HE_time_thresh", "HE RecHit Time Above Threshold",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HEOccupancy=m_dbe->book1D("HE_occupancy","HE RecHit Occupancy",260,-0.5,2599.5);
  h_HEThreshOccupancy=m_dbe->book1D("HE_occupancy_thresh","HE RecHit Occupancy Above Threshold",260,-0.5,2599.5);
  
  // ho
  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/ho");	

  h_HOsizeVsLS=m_dbe->bookProfile("HORecHitsVsLB","HO RecHits vs Luminosity Block",
				  Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				  0,10000);
  h_HOTime=m_dbe->book1D("HO_time","HO RecHit Time",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HOThreshTime=m_dbe->book1D("HO_time_thresh", "HO RecHit Time Above Threshold",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HOOccupancy=m_dbe->book1D("HO_occupancy","HO RecHit Occupancy",217,-0.5,2169.5);
  h_HOThreshOccupancy=m_dbe->book1D("HO_occupancy_thresh","HO RecHit Occupancy Above Threshold",217,-0.5,2169.5);
  
  // hf
  m_dbe->setCurrentFolder(baseFolder_+"/diagnostics/hf");
  h_HFsizeVsLS=m_dbe->bookProfile("HFRecHitsVsLB","HF RecHits vs Luminosity Block",
				  Nlumiblocks_,0.5,Nlumiblocks_+0.5,
				  0,10000);	
  h_HFTime=m_dbe->book1D("HF_time","HF RecHit Time",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HFThreshTime=m_dbe->book1D("HF_time_thresh", "HF RecHit Time Above Threshold",int(RECHITMON_TIME_MAX-RECHITMON_TIME_MIN),RECHITMON_TIME_MIN,RECHITMON_TIME_MAX);
  h_HFOccupancy=m_dbe->book1D("HF_occupancy","HF RecHit Occupancy",173,-0.5,1729.5);
  h_HFThreshOccupancy=m_dbe->book1D("HF_occupancy_thresh","HF RecHit Occupancy Above Threshold",173,-0.5,1729.5);

  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor BEGINRUN -> "<<cpu_timer.cpuTime()<<endl;
   }
  return;
  
} //void HcalRecHitMonitor::setup(...)


/* --------------------------- */

void HcalRecHitMonitor::reset(){}  // reset function is empty for now


/* ------------------------- */

void HcalRecHitMonitor::done()
{
  // Can eventually dump bad rec hit info here, when we decide on a definition for bad rec hits
  return;
  
} // void HcalRecHitMonitor::done()



/* --------------------------------- */

void HcalRecHitMonitor::clearME()
{
  // I don't think this function gets cleared any more.  
  // And need to add code to clear out subfolders as well?
  if (m_dbe)
    {
      m_dbe->setCurrentFolder(baseFolder_);
      m_dbe->removeContents();
    }
  return;
} // void HcalRecHitMonitor::clearME()

/* -------------------------------- */


void HcalRecHitMonitor::processEvent(const HBHERecHitCollection& hbHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits,
				     int  CalibType ,
				     int BCN,
				     const edm::Event & iEvent
				     )
{

  // Check that event is of proper calibration type
  bool processevent=false; 
  if (AllowedCalibTypes_.size()==0)
    processevent=true;
  else
    {
      for (unsigned int i=0;i<AllowedCalibTypes_.size();++i)
	{
	  if (AllowedCalibTypes_[i]==CalibType)
	    {
	      processevent=true;
	      break;
	    }
	}
    }


  /*
    bool passedHLT=false;
  edm::Handle<edm::TriggerResults> hltResults;
  iEvent.getByLabel("HLT",hltResults);
  const edm::TriggerNames & triggerNames = iEvent.triggerNames(*hltResults);
  for (unsigned int i=0;i<hltResults->size();++i)
    {
      string trigName = triggerNames.triggerName(i);
      if (trigName!="HLT_L1_HFtech")
	continue;
      if (hltResults->accept(i))
	{
	  passedHLT=true;
	  break;
	}
    }
  */
  //cout <<"BCN = "<<BCN<<endl;
  //cout <<"PASSED HLT = "<<passedHLT<<endl;

  h_LumiPlot_LS_allevents->Fill(lumiblock);
  h_LumiPlot_BX_allevents->Fill(BCN);

  bool BPTX=false;
  bool passedL1=false;

  if (Online_)
    {
      bool BPTX=false;
      edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
      if (!iEvent.getByLabel("l1GtUnpack",gtRecord))
	{
	  edm::LogWarning("HcalMonitorTasks")<<" HcalRecHitMonitor:  l1GtUnpack L1GlobalTriggerReadoutRecord not found";
	  return;
	}

      bool passedL1=false;
      if (gtRecord.isValid())
	{
	  const DecisionWord dWord = gtRecord->decisionWord();
	  const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();
	  if (tWord.at(4)) BPTX=true;
	  if (tWord.at(8)) passedL1=true;
	  if (tWord.at(9)) passedL1=true;
	  if (tWord.at(10)) passedL1=true;
	}
      //if (passedL1) std::cout <<"PASSED L1 = "<<passedL1<<"  BCN = "<<BCN<<endl;
    }

  if (fVerbosity>1) std::cout <<"<HcalRecHitMonitor::processEvent>  calibType = "<<CalibType<<"  processing event? "<<processevent<<endl;
  if (!processevent)
    return;

  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }


  // increment counters  
  HcalBaseMonitor::processEvent();

  if (hoHits.size()>0) HOpresent_=true;
  if (hfHits.size()>0) HFpresent_=true;

  if (fVerbosity>1) std::cout <<"<HcalRecHitMonitor::processEvent> Processing event..."<<endl;

  processEvent_rechit(hbHits, hoHits, hfHits,passedL1,BPTX,BCN);
  
  // Fill problem cells -- will now fill once per luminosity block
  if (rechit_checkNevents_>0 && ievt_%rechit_checkNevents_ ==0)
    {
      fill_Nevents();
    }
  if (showTiming)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor PROCESSEVENT -> "<<cpu_timer.cpuTime()<<endl;
   }
  return;
} // void HcalRecHitMonitor::processEvent(...)


/* --------------------------------------- */


void HcalRecHitMonitor::processEvent_rechit( const HBHERecHitCollection& hbheHits,
					     const HORecHitCollection& hoHits,
					     const HFRecHitCollection& hfHits,
					     bool passedHLT,
					     bool BPTX,
					     int BCN)
{
  // Gather rechit info
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    }
 
  //const float area[]={0.111,0.175,0.175,0.175,0.175,0.175,0.174,0.178,0.172,0.175,0.178,0.346,0.604};



  if (fVerbosity>1) std::cout <<"<HcalRecHitMonitor::processEvent_rechitenergy> Processing rechits..."<<endl;
  

  // loop over HBHE
  
  int     hbocc=0;
  int     heocc=0;
  int     hboccthresh=0;
  int     heoccthresh=0;

  double en_HFP=0, en_HFM=0, en_HEP=0, en_HEM=0;
  double time_HFP=0, time_HFM=0, time_HEP=0, time_HEM=0;
  double rawtime_HFP=0, rawtime_HFM=0, rawtime_HEP=0, rawtime_HEM=0;
  int hepocc=0, hemocc=0, hfpocc=0, hfmocc=0;

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
      HcalSubdetector subdet = id.subdet();
      int calcEta = CalcEtaBin(subdet,ieta,depth);
      int rbxindex=logicalMap->getHcalFrontEndId(HBHEiter->detid()).rbxIndex();
      int rm= logicalMap->getHcalFrontEndId(HBHEiter->detid()).rm();


      h_HBHE_FlagCorr->Fill(HBHEiter->flagField(HcalCaloFlagLabels::HBHEPulseShape),HBHEiter->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity)); 

      if (subdet==HcalBarrel)
	{
	  HBpresent_=true;
	  if (!checkHB_) continue;
	  
	  //Looping over HB searching for flags --- cris
	  for (int f=0;f<32;f++)
	    {
	      // Let's display HSCP just to see if these bits are set
	      /*
	       if(f == HcalCaloFlagLabels::HSCP_R1R2)
		continue;
              if(f == HcalCaloFlagLabels::HSCP_FracLeader)
                continue;
              if(f == HcalCaloFlagLabels::HSCP_OuterEnergy)
                continue;
              if(f == HcalCaloFlagLabels::HSCP_ExpFit)
                continue;
	      */
	      if (HBHEiter->flagField(f))
		HBflagcounter_[f]++;
	    }

	  if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity))
	    h_FlagMap_HPDMULT->Fill(rbxindex,rm);
	  if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEPulseShape))
	    h_FlagMap_PULSESHAPE->Fill(rbxindex,rm);
	  if (HBHEiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	    h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
	  else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	    h_FlagMap_TIMEADD->Fill(rbxindex,rm);
	  else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	    h_FlagMap_TIMEERROR->Fill(rbxindex,rm);
	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;
	  if (en>=HBenergyThreshold_)
	    {
	      ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	      energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	      time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	    }

	      ++hbocc;
	      if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
		h_HBTime->Fill(ti);
	      else
		++HBtime_[int(ti-RECHITMON_TIME_MIN)];
	      if (en>=HBenergyThreshold_)
		{
		  ++hboccthresh;
		  if (ti<RECHITMON_TIME_MIN || ti>RECHITMON_TIME_MAX)
		    h_HBThreshTime->Fill(ti);
		  else
		    ++HBtime_thresh_[int(ti-RECHITMON_TIME_MIN)];
		} // if (en>=HBenergyThreshold_)
	} // if (id.subdet()==HcalBarrel)

      else if (subdet==HcalEndcap)
	{
	  HEpresent_=true;
	  if (!checkHE_) continue;
	  
	  if (en>collisionHEthresh_ && ieta>0)
	    {
	      en_HEP+=en;
	      time_HEP+=ti*en;
	      rawtime_HEP+=ti;
	      hepocc++;
	    }
	  else if (en>collisionHEthresh_ && ieta<0)
	    {
	      en_HEM+=en;
	      time_HEM+=ti*en;
	      rawtime_HEM+=ti;
	      hemocc++;
	    }

	  //Looping over HE searching for flags --- cris
	  for (int f=0;f<32;f++)
            {
              if (HBHEiter->flagField(f))
                HEflagcounter_[f]++;
            }

	  if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEHpdHitMultiplicity))
	    h_FlagMap_HPDMULT->Fill(rbxindex,rm);
	  if (HBHEiter->flagField(HcalCaloFlagLabels::HBHEPulseShape))
	    h_FlagMap_PULSESHAPE->Fill(rbxindex,rm);
	  if (HBHEiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	    h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
	  else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	    h_FlagMap_TIMEADD->Fill(rbxindex,rm);
	  else if (HBHEiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	    h_FlagMap_TIMEERROR->Fill(rbxindex,rm);


	  ++occupancy_[calcEta][iphi-1][depth-1];
	  energy_[calcEta][iphi-1][depth-1]+=en;
          energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	  time_[calcEta][iphi-1][depth-1]+=ti;
	  if (en>=HEenergyThreshold_)
	    {
	      ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	      energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	      time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	    }
	    {
	      ++heocc;
	      if (ti<-100 || ti>200)
		h_HETime->Fill(ti);
	      else
		++HEtime_[int(ti+100)];
	      if (en>=HEenergyThreshold_)
		{
		  ++heoccthresh;
		  if (ti<-100 || ti>200)
		    h_HEThreshTime->Fill(ti);
		  else
		    ++HEtime_thresh_[int(ti+100)];
		} // if (en>=HEenergyThreshold_)
	    }


	} // else if (id.subdet()==HcalEndcap)
     
    } //for (HBHERecHitCollection::const_iterator HBHEiter=...)

  if (Online_ && passedHLT && BPTX==true)
    {
      if (hepocc >0 && hemocc>0)
	{
	  //cout <<"hepocc = "<<hepocc<<"  hemocc = "<<hemocc<<endl;
	  h_HErawenergydifference->Fill(en_HEP/hepocc-en_HEM/hemocc);
	  h_HErawtimedifference->Fill(rawtime_HEP/hepocc-rawtime_HEM/hemocc);
	}
      // fill overflow, underflow bins if one side unoccupied?  Try it for time plots only right now
      // for now, fill upper, lower bins, not over/underflow
      else if (hepocc>0)
	h_HErawtimedifference->Fill(250000);
      else if (hemocc>0)
	h_HErawtimedifference->Fill(-250000);
      
      if (en_HEP !=0 && en_HEM != 0)
	{
	  h_HEtimedifference->Fill((time_HEP/en_HEP)-(time_HEM/en_HEM));
	  h_HEenergydifference->Fill((en_HEP-en_HEM)/(en_HEP+en_HEM));
	}
    } // if passedHLT

  else if (Online_ && passedHLT && BPTX==false)
    {
      if (hepocc >0 && hemocc>0)
	{
	  //cout <<"hepocc = "<<hepocc<<"  hemocc = "<<hemocc<<endl;
	  h_HEnotBPTXrawenergydifference->Fill(en_HEP/hepocc-en_HEM/hemocc);
	  h_HEnotBPTXrawtimedifference->Fill(rawtime_HEP/hepocc-rawtime_HEM/hemocc);
	}
      // fill overflow, underflow bins if one side unoccupied?  Try it for time plots only right now
      // for now, fill upper, lower bins, not over/underflow
      else if (hepocc>0)
	h_HEnotBPTXrawtimedifference->Fill(250000);
      else if (hemocc>0)
	h_HEnotBPTXrawtimedifference->Fill(-250000);
      
      if (en_HEP !=0 && en_HEM != 0)
	{
	  h_HEnotBPTXtimedifference->Fill((time_HEP/en_HEP)-(time_HEM/en_HEM));
	  h_HEnotBPTXenergydifference->Fill((en_HEP-en_HEM)/(en_HEP+en_HEM));
	}
    } // if passedHLT
  
  ++HB_occupancy_[hbocc/10];
  ++HE_occupancy_[heocc/10];
  ++HB_occupancy_thresh_[hboccthresh/10];
  ++HE_occupancy_thresh_[heoccthresh/10];
  h_HBsizeVsLS->Fill(lumiblock,hbocc);
  h_HEsizeVsLS->Fill(lumiblock,heocc);

  // loop over HO

  if (checkHO_)
   {
     h_HOsizeVsLS->Fill(lumiblock,hoHits.size());
     int hoocc=0;
     int hooccthresh=0;
     for (HORecHitCollection::const_iterator HOiter=hoHits.begin(); HOiter!=hoHits.end(); ++HOiter) 
       { // loop over all hits
	 float en = HOiter->energy();
	 float ti = HOiter->time();

	 HcalDetId id(HOiter->detid().rawId());
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
         int calcEta = CalcEtaBin(HcalOuter,ieta,depth);
	 int rbxindex=logicalMap->getHcalFrontEndId(HOiter->detid()).rbxIndex();
	 int rm= logicalMap->getHcalFrontEndId(HOiter->detid()).rm();
	 
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

	 if (en>=HOenergyThreshold_)
	   {
	     ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	     energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	     time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	   }
	   {
	     ++hoocc;
	     if (ti<-100 || ti>200)
	       h_HOTime->Fill(ti);
	     else
	       ++HOtime_[int(ti+100)];
	     if (en>=HOenergyThreshold_)
	       {
		 ++hooccthresh;
		 if (ti<-100 || ti>200)
		   h_HOThreshTime->Fill(ti);
		 else
		   ++HOtime_thresh_[int(ti+100)];
	       } // if (en>=HOenergyThreshold_)
	   } 
       } // loop over all HO hits
       {
	 ++HO_occupancy_[hoocc/10];
	 ++HO_occupancy_thresh_[hooccthresh/10];
       }

   } // if (checkHO_)
 
  // loop over HF
  if (checkHF_)
   {
     h_HFsizeVsLS->Fill(lumiblock,hfHits.size());

     double EtPlus =0, EtMinus=0;
     double tPlus=0, tMinus=0;
     double ePlus=0, eMinus=0;

     int hfocc=0;
     int hfoccthresh=0;
     for (HFRecHitCollection::const_iterator HFiter=hfHits.begin(); HFiter!=hfHits.end(); ++HFiter) 
       { // loop over all hits
	 float en = HFiter->energy();
	 float ti = HFiter->time();

	 HcalDetId id(HFiter->detid().rawId());
	 
	 int ieta = id.ieta();
	 int iphi = id.iphi();
	 int depth = id.depth();
	 double fEta=0.5*(theHFEtaBounds[abs(ieta)-29]+theHFEtaBounds[abs(ieta)-28]);
         int calcEta = CalcEtaBin(HcalForward,ieta,depth);

	 if (ieta>0)
	   {
	     if (en>0)
	       {
		 tPlus+=en*ti;
		     ePlus+=en;
	       }
	     EtPlus+=en/cosh(fEta);
	   }
	 else if (ieta<0)
	   {
	     if (en>0)
	       {
		 tMinus+=en*ti;
		 eMinus+=en;
	       }
		 EtMinus+=en/cosh(fEta);
	   }

	 if (en>collisionHFthresh_ && ieta>0)
	    {
	      en_HFP+=en;
	      time_HFP+=ti*en;
	      rawtime_HFP+=ti;
	      hfpocc++;
	    }
	 else if (en>collisionHFthresh_ && ieta<0)
	   {
	     en_HFM+=en;
	     time_HFM+=ti*en;
	     rawtime_HFM+=ti;
	     hfmocc++;
	    }
	 
	 int rbxindex=logicalMap->getHcalFrontEndId(HFiter->detid()).rbxIndex();
	 int rm= logicalMap->getHcalFrontEndId(HFiter->detid()).rm(); 
	 
	 h_HF_FlagCorr->Fill(HFiter->flagField(HcalCaloFlagLabels::HFDigiTime),HFiter->flagField(HcalCaloFlagLabels::HFLongShort)); 
	 if (HFiter->flagField(HcalCaloFlagLabels::TimingSubtractedBit))
	   h_FlagMap_TIMESUBTRACT->Fill(rbxindex,rm);
	 else if (HFiter->flagField(HcalCaloFlagLabels::TimingAddedBit))
	   h_FlagMap_TIMEADD->Fill(rbxindex,rm);
	 else if (HFiter->flagField(HcalCaloFlagLabels::TimingErrorBit))
	   h_FlagMap_TIMEERROR->Fill(rbxindex,rm);

	 if (HFiter->flagField(HcalCaloFlagLabels::HFDigiTime))
	   h_FlagMap_DIGITIME->Fill(rbxindex,rm);
	 if (HFiter->flagField(HcalCaloFlagLabels::HFLongShort))
	   h_FlagMap_LONGSHORT->Fill(rbxindex,rm);

	 //Looping over HF searching for flags --- cris
	 for (int f=0;f<32;f++)
	   {
	     if (HFiter->flagField(f))
	       HFflagcounter_[f]++;
	   }


	 ++occupancy_[calcEta][iphi-1][depth-1];
	 energy_[calcEta][iphi-1][depth-1]+=en;
         energy2_[calcEta][iphi-1][depth-1]+=pow(en,2);
	 time_[calcEta][iphi-1][depth-1]+=ti;


	 //if (en/cosh(etaBounds[abs(ieta)-29]/area[abs(ieta)-29])>=HFenergyThreshold_)
	 if (en>=HFenergyThreshold_)
	   {
	     ++occupancy_thresh_[calcEta][iphi-1][depth-1];
	     energy_thresh_[calcEta][iphi-1][depth-1]+=en;
	     time_thresh_[calcEta][iphi-1][depth-1]+=ti;
	   }
	   {
	     ++hfocc;
	     if (ti<-100 || ti>200)
	       h_HFTime->Fill(ti);
	     else
	       ++HFtime_[int(ti+100)];
	     if (en>=HFenergyThreshold_)
	       {
		 ++hfoccthresh;
		 if (ti<-100 || ti>200)
		   h_HFThreshTime->Fill(ti);
		 else
		   ++HFtime_thresh_[int(ti+100)];
	       } // if (en>=HFenergyThreshold_)
	   }
       } // loop over all HF hits
     

     if (ePlus>0) tPlus/=ePlus;
     if (eMinus>0) tMinus/=eMinus;
     
     double mintime=min(tPlus,tMinus);
     double minHT=min(EtMinus,EtPlus);
     h_LumiPlot_MinTime_vs_MinHT->Fill(minHT, mintime);

     h_LumiPlot_SumHT_HFPlus_vs_HFMinus->Fill(EtMinus,EtPlus);
     h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus->Fill(eMinus,ePlus);
     h_LumiPlot_timeHFPlus_vs_timeHFMinus->Fill(tMinus,tPlus);

     if (tPlus>25 && tMinus>25 &&  EtMinus>1 && EtPlus>1)
       {
	 h_LumiPlot_EventsPerLS->Fill(lumiblock);
	 h_LumiPlot_BX_goodevents->Fill(BCN);
       }
     if (EtMinus>1 && EtPlus>1)
       {
	 h_LumiPlot_EventsPerLS_notimecut->Fill(lumiblock);
	 h_LumiPlot_BX_goodevents_notimecut->Fill(BCN);
       }

     //if (hfpocc > 0 && hfmocc>0)
     // cout <<"HF time difference = "<<rawtime_HFP/hfpocc <<" - "<<rawtime_HFM/hfmocc<<" = "<<(rawtime_HFP/hfpocc-rawtime_HFM/hfmocc)<<endl;
     if (Online_ && passedHLT && BPTX==true)
       {
	 if (hfpocc >0 && hfmocc>0)
	   {
	     h_HFrawenergydifference->Fill(en_HFP/hfpocc-en_HFM/hfmocc);
	     h_HFrawtimedifference->Fill(rawtime_HFP/hfpocc-rawtime_HFM/hfmocc);
	   }
	 // fill overflow, underflow bins if one side unoccupied?  Try it for time plots only right now
	 else if (hfpocc>0)
	   h_HFrawtimedifference->Fill(2500);
	 else if (hfmocc>0)
	   h_HFrawtimedifference->Fill(-2500);
	 
	 //cout <<"HF occ + = "<<hfpocc<<"  - = "<<hfmocc<<endl;
	 if (en_HFP !=0 && en_HFM != 0)
	   {
	     h_HFtimedifference->Fill((time_HFP/en_HFP)-(time_HFM/en_HFM));
	     h_HFenergydifference->Fill((en_HFP-en_HFM)/(en_HFP+en_HFM));
	   }
       } // if (passedHLT)

     else if (Online_ && passedHLT && BPTX==false)
       {
	 if (hfpocc >0 && hfmocc>0)
	   {
	     h_HFnotBPTXrawenergydifference->Fill(en_HFP/hfpocc-en_HFM/hfmocc);
	     h_HFnotBPTXrawtimedifference->Fill(rawtime_HFP/hfpocc-rawtime_HFM/hfmocc);
	   }
	 // fill overflow, underflow bins if one side unoccupied?  Try it for time plots only right now
	 else if (hfpocc>0)
	   h_HFnotBPTXrawtimedifference->Fill(2500);
	 else if (hfmocc>0)
	   h_HFnotBPTXrawtimedifference->Fill(-2500);
	 
	 //cout <<"HF occ + = "<<hfpocc<<"  - = "<<hfmocc<<endl;
	 if (en_HFP !=0 && en_HFM != 0)
	   {
	     h_HFnotBPTXtimedifference->Fill((time_HFP/en_HFP)-(time_HFM/en_HFM));
	     h_HFnotBPTXenergydifference->Fill((en_HFP-en_HFM)/(en_HFP+en_HFM));
	   }
       } // passsed HLT, !101

     ++HF_occupancy_[hfocc/10];
     ++HF_occupancy_thresh_[hfoccthresh/10];
       
    } // if (checkHF_)
   
 if (showTiming)
   {
     cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor PROCESSEVENT_RECHITENERGY -> "<<cpu_timer.cpuTime()<<endl;
   }
 return;
} // void HcalRecHitMonitor::processEvent_rechitenergy

/* --------------------------------------- */


void HcalRecHitMonitor::endLuminosityBlock()
{
  if (LBprocessed_==true) return; // filling already took place this LB
  fill_Nevents();
  LBprocessed_=true;
  return;
} //endLuminosityBlock

void HcalRecHitMonitor::fill_Nevents(void)
{
  if (showTiming)
    {
      cpu_timer.reset(); cpu_timer.start();
    } 

  // looking at the contents of HbFlagcounters
  if (fVerbosity>0)
    {
      for (int k = 0; k < 32; k++){
	std::cout << "<HcalRecHitMonitor::fill_Nevents>  HF Flag counter:  Bin #" << k+1 << " = "<< HFflagcounter_[k] << endl;
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
  if (ievt_>0)
    {
      for (int mydepth=0;mydepth<4;++mydepth)
	{
	  for (int eta=0;eta<OccupancyByDepth.depth[mydepth]->getNbinsX();++eta)
	    {
	      for (int phi=0;phi<72;++phi)
		{
		  OccupancyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_[eta][phi][mydepth]);
		  OccupancyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,occupancy_thresh_[eta][phi][mydepth]);
		  SumEnergyByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_[eta][phi][mydepth]);
                  SqrtSumEnergy2ByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,sqrt(energy2_[eta][phi][mydepth]));
		  SumEnergyThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,energy_thresh_[eta][phi][mydepth]);
		  SumTimeByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_[eta][phi][mydepth]);
		  SumTimeThreshByDepth.depth[mydepth]->setBinContent(eta+1,phi+1,time_thresh_[eta][phi][mydepth]);
		} // for (int phi=0;phi<72;++phi)
	    } // for (int eta=0;eta<OccupancyByDepth...;++eta)
	} // for (int mydepth=0;...)

      FillUnphysicalHEHFBins(OccupancyByDepth);
      FillUnphysicalHEHFBins(OccupancyThreshByDepth);
      FillUnphysicalHEHFBins(SumEnergyByDepth);
      FillUnphysicalHEHFBins(SqrtSumEnergy2ByDepth);
      FillUnphysicalHEHFBins(SumEnergyThreshByDepth);
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

  if (fVerbosity>0)
    std::cout <<"<HcalRecHitMonitor::fill_Nevents> FILLED REC HIT CELL PLOTS"<<endl;
    
  if (showTiming)
    {
      cpu_timer.stop();  std::cout <<"TIMER:: HcalRecHitMonitor FILLNEVENTS -> "<<cpu_timer.cpuTime()<<endl;
    }

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
  for (int i=0;i<ETABINS;++i)
    {
      for (int j=0;j<PHIBINS;++j)
	{
	  for (int k=0;k<6;++k)
	    {
	      occupancy_[i][j][k]=0;
	      occupancy_thresh_[i][j][k]=0;
	      energy_[i][j][k]=0;
              energy2_[i][j][k]=0;
	      energy_thresh_[i][j][k]=0;
	      time_[i][j][k]=0;
	      time_thresh_[i][j][k]=0;
	    }
	} // for (int j=0;j<PHIBINS;++j)
    } // for (int i=0;i<ETABINS;++i)

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
      if (i<=260)
	{
	  HB_occupancy_[i]=0;
	  HE_occupancy_[i]=0;
	  HB_occupancy_thresh_[i]=0;
	  HE_occupancy_thresh_[i]=0;
	}
      if (i<=217)
	{
	  HO_occupancy_[i]=0;
	  HO_occupancy_thresh_[i]=0;
	}
      if (i<=173)
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
