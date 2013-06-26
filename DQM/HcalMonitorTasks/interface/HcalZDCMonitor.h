#ifndef DQM_HCALMONITORTASKS_HCALZDCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALZDCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"
#include <cmath>

/** \class HcalZDCMonitor
 *
 * $DATE: 2010/02/04
 * $Revision:
 * \author S.Sen
 */

class HcalZDCMonitor: public HcalBaseMonitor
{
 public:
  HcalZDCMonitor();
  ~HcalZDCMonitor();
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const ZDCDigiCollection& digi,
                    const ZDCRecHitCollection& rechit);
  //void done();
  void reset();
  void endLuminosityBlock(void);
 private:
  //virtual void endJob();
  //void endLuminosityBlock(void);
  //void zeroCounters();
  //void fillHistos();
  //void setZDClabels(MonitorElement* h);
  double getTime(std::vector<double> fData, unsigned int ts_min, unsigned int ts_max, double& fSum);
  //int getTSMax(std::vector<double> fData);
  //bool isGood(std::vector<double> fData, double fCut, double fPercentage);

  bool checkZDC_;
  int NumBadZDC;
  MonitorElement* ProblemsVsLB_ZDC;

  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;
  HcalCalibrations calibs_;
  int ievt_;
  //int zdc_checkNevents_;
  MonitorElement* meEVT_;
  MonitorElement* h_2D_saturation;
  MonitorElement* h_2D_charge;
  MonitorElement* h_2D_TSMean;
  MonitorElement* h_2D_RecHitEnergy;
  MonitorElement* h_2D_RecHitTime;
  MonitorElement* h_ZDCP_EM_Pulse[5];
  MonitorElement* h_ZDCM_EM_Pulse[5];
  MonitorElement* h_ZDCP_EM_Charge[5];
  MonitorElement* h_ZDCM_EM_Charge[5];
  MonitorElement* h_ZDCP_EM_TSMean[5];
  MonitorElement* h_ZDCM_EM_TSMean[5];
  MonitorElement* h_ZDCP_HAD_Pulse[4];
  MonitorElement* h_ZDCM_HAD_Pulse[4];
  MonitorElement* h_ZDCP_HAD_Charge[4];
  MonitorElement* h_ZDCM_HAD_Charge[4];
  MonitorElement* h_ZDCP_HAD_TSMean[4];
  MonitorElement* h_ZDCM_HAD_TSMean[4];
  MonitorElement* h_ZDCP_EM_RecHitEnergy[5];
  MonitorElement* h_ZDCM_EM_RecHitEnergy[5];
  MonitorElement* h_ZDCP_EM_RecHitTiming[5];
  MonitorElement* h_ZDCM_EM_RecHitTiming[5];
  MonitorElement* h_ZDCP_HAD_RecHitEnergy[4];
  MonitorElement* h_ZDCM_HAD_RecHitEnergy[4];
  MonitorElement* h_ZDCP_HAD_RecHitTiming[4];
  MonitorElement* h_ZDCM_HAD_RecHitTiming[4];
};

#endif
