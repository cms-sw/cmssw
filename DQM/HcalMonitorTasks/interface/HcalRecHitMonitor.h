#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h" // for eta bounds
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>

#define RECHITMON_TIME_MIN -250
#define RECHITMON_TIME_MAX 250

/** \class HcalRecHitMonitor
  *
  * $Date: 2010/02/18 20:42:07 $
  * $Revision: 1.45 $
  * \author J. Temple - Univ. of Maryland
  */


class HcalRecHitMonitor: public HcalBaseMonitor {

 public:
  HcalRecHitMonitor();

  ~HcalRecHitMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void beginRun();
  void done();
  void clearME(); // overrides base class function
  void reset();
  void zeroCounters();
 
  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    int CalibType,
		    int BCN,
		    const edm::Event& iEvent
		    );

  void processEvent_rechit( const HBHERecHitCollection& hbheHits,
			    const HORecHitCollection& hoHits,
			    const HFRecHitCollection& hfHits,
			    bool passedHLT,
			    bool BPTX,
			    int BCN);
			    

  void endLuminosityBlock();
 private:
  
  void fill_Nevents();

  int rechit_checkNevents_;  // specify how often to fill histograms

  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;

  double rechit_minErrorFlag_; // minimum error rate needed to dump out bad bin info 

  HcalLogicalMap* logicalMap;

  // Basic Histograms
  EtaPhiHists OccupancyByDepth;
  EtaPhiHists OccupancyThreshByDepth;

  EtaPhiHists SumEnergyByDepth;
  EtaPhiHists SqrtSumEnergy2ByDepth;
  EtaPhiHists SumEnergyThreshByDepth;
  EtaPhiHists SumTimeByDepth;
  EtaPhiHists SumTimeThreshByDepth;


  unsigned int occupancy_[85][72][4]; // will get filled when rechit found
  unsigned int occupancy_thresh_[85][72][4]; // filled when above given energy
  double energy_[85][72][4]; // will get filled when rechit found
  double energy2_[85][72][4]; // will get filled when rechit found
  double energy_thresh_[85][72][4]; // filled when above given  
  double time_[85][72][4]; // will get filled when rechit found
  double time_thresh_[85][72][4]; // filled when above given energy

  double HBtime_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HBtime_thresh_[RECHITMON_TIME_MAX-RECHITMON_TIME_MIN];
  double HB_occupancy_[261];
  double HB_occupancy_thresh_[261];
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

  MonitorElement* h_HF_FlagCorr;
  MonitorElement* h_HBHE_FlagCorr;

  double collisionHFthresh_;
  double collisionHEthresh_;

  MonitorElement* h_HFtimedifference;
  MonitorElement* h_HFenergydifference;
  MonitorElement* h_HEtimedifference;
  MonitorElement* h_HEenergydifference;
  MonitorElement* h_HFrawenergydifference;
  MonitorElement* h_HErawenergydifference;
  MonitorElement* h_HFrawtimedifference;
  MonitorElement* h_HErawtimedifference;

  MonitorElement* h_HFnotBPTXtimedifference;
  MonitorElement* h_HFnotBPTXenergydifference;
  MonitorElement* h_HEnotBPTXtimedifference;
  MonitorElement* h_HEnotBPTXenergydifference;
  MonitorElement* h_HFnotBPTXrawenergydifference;
  MonitorElement* h_HEnotBPTXrawenergydifference;
  MonitorElement* h_HFnotBPTXrawtimedifference;
  MonitorElement* h_HEnotBPTXrawtimedifference;

  MonitorElement* h_LumiPlot_LS_allevents;
  MonitorElement* h_LumiPlot_EventsPerLS;
  MonitorElement* h_LumiPlot_EventsPerLS_notimecut;

  MonitorElement* h_LumiPlot_SumHT_HFPlus_vs_HFMinus;
  MonitorElement* h_LumiPlot_timeHFPlus_vs_timeHFMinus;

  MonitorElement* h_LumiPlot_SumEnergy_HFPlus_vs_HFMinus;
  
  MonitorElement* h_LumiPlot_BX_allevents;
  MonitorElement* h_LumiPlot_BX_goodevents;
  MonitorElement* h_LumiPlot_BX_goodevents_notimecut;

  MonitorElement* h_LumiPlot_MinTime_vs_MinHT;
  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
};

#endif
