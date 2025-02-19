#ifndef DQM_HCALMONITORTASKS_HCALCALOTOWERMONITOR_H
#define DQM_HCALMONITORTASKS_HCALCALOTOWERMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include <map>

/** \class HcalCaloTowerMonitor
  *  
  * $Date: 2008/08/14 18:40:28 $
  * $Revision: 1.2 $
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

   // calotower histograms
   MonitorElement* caloTowerOccMap;
   MonitorElement* caloTowerEnergyMap; 
   MonitorElement* caloTowerTime;
   MonitorElement* caloTowerEnergy;
   MonitorElement* caloTowerMeanEnergyEta;

   // hcal histograms
   MonitorElement* hcalOccMap;
   MonitorElement* hcalEnergyMap;
   MonitorElement* hcalTime;
   MonitorElement* hcalEnergy;
   MonitorElement* hcalMeanEnergyEta;

   // ecal histograms
   MonitorElement* ecalOccMap;
   MonitorElement* ecalEnergyMap;
   MonitorElement* ecalTime;
   MonitorElement* ecalEnergy;
   MonitorElement* ecalMeanEnergyEta;
   
   // comparison plots
   MonitorElement* time_HcalvsEcal;
   MonitorElement* time_CaloTowervsEcal;
   MonitorElement* time_CaloTowervsHcal;
   MonitorElement* energy_HcalvsEcal;
   MonitorElement* energy_CaloTowervsEcal;
   MonitorElement* energy_CaloTowervsHcal;

   MonitorElement* meEVT_;


};

#endif
