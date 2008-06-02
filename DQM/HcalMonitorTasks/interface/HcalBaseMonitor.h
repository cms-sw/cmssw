#ifndef DQM_HCALMONITORTASKS_HCALBASEMONITOR_H
#define DQM_HCALMONITORTASKS_HCALBASEMONITOR_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <iostream>

using namespace std;
/** \class HcalBaseMonitor
  *  
  * $Date: 2006/02/02 16:32:12 $
  * $Revision: 1.4 $
  * \author W. Fisher - FNAL
  */
class HcalBaseMonitor {
public:
  HcalBaseMonitor(); 
  virtual ~HcalBaseMonitor(); 

  virtual void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  virtual void done();

  void setVerbosity(int verb) { fVerbosity = verb; }
  int getVerbosity() const { return fVerbosity; }
  bool vetoCell(HcalDetId id);

protected:
  
  int fVerbosity;
  DaqMonitorBEInterface* m_dbe;
  vector<string> hotCells_;

};

#endif
