/*
 * $Date: 2008/01/12 23:51:50 $
 * $Revision: 1.2 $
 * $Author: strang $
*/

#ifndef DaqMonitorROOTBackEnd_h
#define DaqMonitorROOTBackEnd_h

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

class DaqMonitorROOTBackEnd : public DaqMonitorBEInterface
{  
  public:
  
  DaqMonitorROOTBackEnd(edm::ParameterSet const& pset);  
  
  virtual ~DaqMonitorROOTBackEnd(){} ;
  
};

#endif
