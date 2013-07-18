#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#define RECHITMON_TIME_MIN -250
#define RECHITMON_TIME_MAX 250

/** \class HcalRecHitMonitor
  *
  * $Date: 2012/06/27 13:20:29 $
  * $Revision: 1.52 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalRecHitMonitor: public HcalBaseDQMonitor {

 public:
  HcalRecHitMonitor(const edm::ParameterSet& ps);

  ~HcalRecHitMonitor();

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			  const edm::EventSetup& c);
  void endJob();
  void cleanup();
  void reset();
  void zeroCounters();
 
  void analyze(const edm::Event&, const edm::EventSetup&);

  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    int BCN,
		    const edm::Event& iEvent
		    );

  void processEvent_rechit( const HBHERecHitCollection& hbheHits,
			    const HORecHitCollection& hoHits,
			    const HFRecHitCollection& hfHits,
			    bool passedHcalHLT,
			    bool passedMinBiasHLT,
			    int BCN);
 private:
  
  void fill_Nevents();

  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;

  double ETThreshold_;
  double HBETThreshold_;
  double HEETThreshold_;
  double HOETThreshold_;
  double HFETThreshold_;

  // Basic Histograms
  EtaPhiHists OccupancyByDepth;
  EtaPhiHists OccupancyThreshByDepth;

  EtaPhiHists SumEnergyByDepth;
  EtaPhiHists SqrtSumEnergy2ByDepth;
  EtaPhiHists SumEnergyThreshByDepth;
  EtaPhiHists SqrtSumEnergy2ThreshByDepth;
  EtaPhiHists SumTimeByDepth;
  EtaPhiHists SumTimeThreshByDepth;

  unsigned int occupancy_[85][72][4]; // will get filled when rechit found
  unsigned int occupancy_thresh_[85][72][4]; // filled when above given energy
  double energy_[85][72][4]; // will get filled when rechit found
  double energy2_[85][72][4]; // will get filled when rechit found
  double energy_thresh_[85][72][4]; // filled when above given  
  double energy2_thresh_[85][72][4]; // filled when above given
  double time_[85][72][4]; // will get filled when rechit found
  double time_thresh_[85][72][4]; // filled when above given energy

  double HBtime_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HBtime_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HB_occupancy_[260];
  double HB_occupancy_thresh_[260];
  double HEtime_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HEtime_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HE_occupancy_[260];
  double HE_occupancy_thresh_[260];
  double HOtime_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HOtime_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HO_occupancy_[218];
  double HO_occupancy_thresh_[218];
  double HFtime_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HFtime_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HFenergyLong_[200];
  double HFenergyLong_thresh_[200];
  double HFtimeLong_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HFtimeLong_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HFenergyShort_[200];
  double HFenergyShort_thresh_[200];
  double HFtimeShort_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HFtimeShort_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HF_occupancy_[174];
  double HF_occupancy_thresh_[174];
  double HFlong_occupancy_[865];
  double HFlong_occupancy_thresh_[865];
  double HFshort_occupancy_[865];
  double HFshort_occupancy_thresh_[865];

  int HEflagcounter_[32];
  int HBflagcounter_[32];
  int HOflagcounter_[32];
  int HFflagcounter_[32];

  // Diagnostic plots

  MonitorElement* h_rechitieta;
  MonitorElement* h_rechitiphi;

  MonitorElement* h_rechitieta_05;
  MonitorElement* h_rechitieta_10;
  MonitorElement* h_rechitieta_25;
  MonitorElement* h_rechitieta_100;
  MonitorElement* h_rechitiphi_05;
  MonitorElement* h_rechitiphi_10;
  MonitorElement* h_rechitiphi_25;
  MonitorElement* h_rechitiphi_100;

  MonitorElement* h_rechitieta_thresh;
  MonitorElement* h_rechitiphi_thresh;

  MonitorElement* h_HBsizeVsLS;
  MonitorElement* h_HEsizeVsLS;
  MonitorElement* h_HOsizeVsLS;
  MonitorElement* h_HFsizeVsLS;

  MonitorElement* h_HBTime;
  MonitorElement* h_HBThreshTime;
  MonitorElement* h_HBOccupancy;
  MonitorElement* h_HBThreshOccupancy;

  MonitorElement* h_HETime;
  MonitorElement* h_HEThreshTime;
  MonitorElement* h_HEOccupancy;
  MonitorElement* h_HEThreshOccupancy;

  MonitorElement* h_HOTime;
  MonitorElement* h_HOThreshTime;
  MonitorElement* h_HOOccupancy;
  MonitorElement* h_HOThreshOccupancy;

  MonitorElement* h_HFTime;
  MonitorElement* h_HFThreshTime;
  MonitorElement* h_HFOccupancy;
  MonitorElement* h_HFThreshOccupancy;

  MonitorElement* h_HBflagcounter;
  MonitorElement* h_HEflagcounter;
  MonitorElement* h_HOflagcounter;
  MonitorElement* h_HFflagcounter;
  
  MonitorElement* h_FlagMap_HPDMULT;
  MonitorElement* h_FlagMap_PULSESHAPE;
  MonitorElement* h_FlagMap_DIGITIME;
  MonitorElement* h_FlagMap_LONGSHORT;
  MonitorElement* h_FlagMap_TIMEADD;
  MonitorElement* h_FlagMap_TIMESUBTRACT;
  MonitorElement* h_FlagMap_TIMEERROR;
                 
  MonitorElement* h_HFLongShort_vs_LS;
  MonitorElement* h_HFDigiTime_vs_LS;
  MonitorElement* h_HBHEHPDMult_vs_LS;
  MonitorElement* h_HBHEPulseShape_vs_LS;

  MonitorElement* h_HF_FlagCorr;
  MonitorElement* h_HBHE_FlagCorr;

  double timediffThresh_;

  MonitorElement* h_HFtimedifference;
  MonitorElement* h_HFenergydifference;
  MonitorElement* h_HEtimedifference;
  MonitorElement* h_HEenergydifference;

  MonitorElement* h_HF_HcalHLT_weightedtimedifference;
  MonitorElement* h_HF_HcalHLT_energydifference;
  MonitorElement* h_HE_HcalHLT_weightedtimedifference;
  MonitorElement* h_HE_HcalHLT_energydifference;

  MonitorElement* h_LumiPlot_LS_allevents;
  MonitorElement* h_LumiPlot_LS_MinBiasEvents;
  MonitorElement* h_LumiPlot_LS_MinBiasEvents_notimecut;
  MonitorElement* h_LumiPlot_LS_HcalHLTEvents;
  MonitorElement* h_LumiPlot_LS_HcalHLTEvents_notimecut;

  MonitorElement* h_LumiPlot_SumHT_HFPlus_vs_HFMinus;
  MonitorElement* h_LumiPlot_timeHFPlus_vs_timeHFMinus;

  MonitorElement* h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus;
  
  MonitorElement* h_LumiPlot_BX_allevents;
  MonitorElement* h_LumiPlot_BX_MinBiasEvents;
  MonitorElement* h_LumiPlot_BX_MinBiasEvents_notimecut;
  MonitorElement* h_LumiPlot_BX_HcalHLTEvents;
  MonitorElement* h_LumiPlot_BX_HcalHLTEvents_notimecut;

  MonitorElement* h_LumiPlot_MinTime_vs_MinHT;
  MonitorElement* h_LumiPlot_timeHT_HFM;
  MonitorElement* h_LumiPlot_timeHT_HFP;

  MonitorElement* h_TriggeredEvents;
  MonitorElement* h_HFP_weightedTime;
  MonitorElement* h_HFM_weightedTime;
  MonitorElement* h_HEP_weightedTime;
  MonitorElement* h_HEM_weightedTime;
  MonitorElement* h_HBP_weightedTime;
  MonitorElement* h_HBM_weightedTime;

  MonitorElement* h_HBTimeVsEnergy;
  MonitorElement* h_HETimeVsEnergy;
  MonitorElement* h_HOTimeVsEnergy;
  MonitorElement* h_HFTimeVsEnergy;
  MonitorElement* HFP_HFM_Energy;

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
  bool setupDone_;
  
  edm::InputTag hbheRechitLabel_, hoRechitLabel_, hfRechitLabel_;
  edm::InputTag l1gtLabel_;
  edm::InputTag hltresultsLabel_;
  std::vector <std::string> HcalHLTBits_;
  std::vector <std::string> MinBiasHLTBits_;
};

#endif
