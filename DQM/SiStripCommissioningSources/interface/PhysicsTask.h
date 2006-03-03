#ifndef DQM_SiStripCommissioningSources_PhysicsTask_h
#define DQM_SiStripCommissioningSources_PhysicsTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"

/**
   \class PhysicsTask
*/
class PhysicsTask : public CommissioningTask {
  
 public:
  
  PhysicsTask( DaqMonitorBEInterface*, const SiStripModule& );
  virtual ~PhysicsTask();
  
 private:

  virtual void book( const SiStripModule& );
  virtual void fill( const vector<StripDigi>& );
  virtual void update();
  
  HistoSet landau_;

};

#endif // DQM_SiStripCommissioningSources_PhysicsTask_h


