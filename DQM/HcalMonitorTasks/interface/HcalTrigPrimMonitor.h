#ifndef DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H
#define DQM_HCALMONITORTASKS_HCALTRIGPRIMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include <set>

/** \class HcalTrigPrimMonitor
  *  
  * $Date: 2009/08/24 11:22:14 $
  * $Revision: 1.18 $
  * \author W. Fisher - FNAL
  */

class HcalTrigPrimMonitor: public HcalBaseMonitor {
 public:
  HcalTrigPrimMonitor(); 
  ~HcalTrigPrimMonitor(); 
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbhedigi,
		    const HODigiCollection& hodigi,
		    const HFDigiCollection& hfdigi,		    
		    const HcalTrigPrimDigiCollection& tpDigis,
		    const HcalTrigPrimDigiCollection& emultpDigis,
                const FEDRawDataCollection& rawraw,
		    const HcalElectronicsMap& emap);
  void clearME();
  void reset();


private:

  void buildFrontEndErrMap(const FEDRawDataCollection& rawraw, const HcalElectronicsMap& emap);
  HcalTrigTowerGeometry theTrigTowerGeometry;
  std::set<uint32_t> FrontEndErrors;
  // Alarm threshold for ZS run
  int ZSAlarmThreshold_;
  //Error flag per event
  // 0 - HBHE
  // 1 - HF
  unsigned int ErrorFlagPerEvent_[2];
  unsigned int ErrorFlagPerEventZS_[2];

  enum ErrorFlag{
    kZeroTP=-1,
    kMatched,
    kMismatchedEt,
    kMismatchedFG,
    kDataOnly,
    kEmulOnly,
    kMissingData,
    kMissingEmul,
    kNErrorFlag,
    kUnknown = kNErrorFlag
  };

  // Summary
  // 0 - HBHE
  // 1 - HF
  MonitorElement* Summary_;
  MonitorElement* SummaryZS_;
  MonitorElement* ErrorFlagSummary_;
  MonitorElement* ErrorFlagSummaryZS_;
  MonitorElement* EtCorr_[2];

  // TP Occupancy
  MonitorElement* TPOccupancy_;
  MonitorElement* TPOccupancyEta_;
  MonitorElement* TPOccupancyPhi_;

  MonitorElement* NonZeroTP_;
  MonitorElement* MatchedTP_;
  MonitorElement* MismatchedEt_;
  MonitorElement* MismatchedFG_;
  MonitorElement* DataOnly_;    
  MonitorElement* EmulOnly_;
  MonitorElement* MissingData_;
  MonitorElement* MissingEmul_;

  // Energy Plots
  // 0 - HBHE
  // 1 - HF
  MonitorElement* EnergyPlotsAllData_[2];
  MonitorElement* EnergyPlotsAllEmul_[2];
  MonitorElement* EnergyPlotsMismatchedFG_[2];
  MonitorElement* EnergyPlotsDataOnly_[2];
  MonitorElement* EnergyPlotsEmulOnly_[2];
  MonitorElement* EnergyPlotsMissingData_[2];
  MonitorElement* EnergyPlotsMissingEmul_[2];

};
#endif
