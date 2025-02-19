#ifndef DQM_CASTORMONITOR_CASTOREVENTDISPLAY_H
#define DQM_CASTORMONITOR_CASTOREVENTDISPLAY_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include <string>
#include <map>
#include <vector>


class CastorEventDisplay: public CastorBaseMonitor {
public:
  CastorEventDisplay(); 
  ~CastorEventDisplay(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorRecHitCollection& castorHits, const CaloGeometry& caloGeometry );
  void reset();

private:  


 
  int ievt_;
  double X_pos;
  double Y_pos;
  double Z_pos;
  double X_pos_maxE;
  double Y_pos_maxE;
  double Z_pos_maxE;

  bool offline_;
  float  energy;
  float  allEnergyEvent;
  float  maxEnergyEvent;

  ////---- define Monitoring elements
  MonitorElement* meCastor3Dhits; //-- cumulative event display
  MonitorElement* meCastor3DhitsMaxEnergy; //-- dispay of an event with the largest deposited energy
  MonitorElement* meEVT_;

};

#endif
