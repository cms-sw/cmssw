#ifndef DQM_SiStripCommissioningSources_CommissioningTask_H
#define DQM_SiStripCommissioningSources_CommissioningTask_H

#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include <vector> 

class DaqMonitorBEInterface;
class SiStripModule;
class MonitorElement;

using namespace std;

/**
   @class CommissioningTask
*/
class CommissioningTask {

 public:

  struct HistoSet {
    MonitorElement* meSumOfSquares_;
    MonitorElement* meSumOfContents_;
    MonitorElement* meNumOfEntries_;
    vector<unsigned int> vSumOfSquares_;
    vector<unsigned int> vSumOfContents_;
    vector<unsigned int> vNumOfEntries_;
  };
  
  CommissioningTask( DaqMonitorBEInterface*, const SiStripModule& );
  virtual ~CommissioningTask();
  
  virtual void fillHistograms( const vector<StripDigi>& ); //@@ requires "SiStripEventInfo" info as well!
  void updateFreq( int freq ) { updateFreq_ = freq; }
  
 protected:
  
  DaqMonitorBEInterface* dqm_;
  unsigned int updateFreq_;
  unsigned int fillCntr_;
  
 private:
  
  CommissioningTask() {;}
  
  virtual void book( const SiStripModule& );
  virtual void fill( const vector<StripDigi>& ) = 0; //@@ requires "SiStripEventInfo" info as well!
  virtual void update() = 0;
  
};

#endif // DQM_SiStripCommissioningSources_CommissioningTask_H

