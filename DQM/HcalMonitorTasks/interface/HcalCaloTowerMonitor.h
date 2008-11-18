#ifndef DQM_HCALMONITORTASKS_HCALCALOTOWERMONITOR_H
#define DQM_HCALMONITORTASKS_HCALCALOTOWERMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <map>

/** \class HcalCaloTowerMonitor
  *  
  * $Date: 2008/03/05 09:21:45 $
  * $Revision: 1.1 $
  * \author J. Temple - Univ. of Maryland
  */


class HcalCaloTowerMonitor: public HcalBaseMonitor {
 public:
  HcalCaloTowerMonitor(); 
  ~HcalCaloTowerMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CaloTowerCollection& calotower);
  void reset();
  int getIeta(double eta); // get ieta index, given physical coordinate eta
 private:  ///Monitoring elements

   bool debug_;
   double etaMax_, etaMin_, phiMax_, phiMin_;
   int etaBins_, phiBins_;
   int ievt_;

   MonitorElement* caloTowerOcc;
   MonitorElement* caloTowerEnergy; 
   MonitorElement* meEVT_;


};

#endif
