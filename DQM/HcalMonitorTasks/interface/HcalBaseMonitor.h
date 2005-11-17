#ifndef DQM_HCALMONITORTASKS_HCALBASEMONITOR_H
#define DQM_HCALMONITORTASKS_HCALBASEMONITOR_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CondFormats/HcalMapping/interface/HcalMappingTextFileReader.h"
#include <iostream>

using namespace std;
/** \class HcalBaseMonitor
  *  
  * $Date: 2005/11/14 16:48:07 $
  * $Revision: 1.2 $
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

protected:
  
  int fVerbosity;
  DaqMonitorBEInterface* m_dbe;
  HcalMapping m_readoutMap;
  string m_readoutMapSource;

};

#endif
