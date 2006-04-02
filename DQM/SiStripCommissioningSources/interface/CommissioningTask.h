#ifndef DQM_SiStripCommissioningSources_CommissioningTask_H
#define DQM_SiStripCommissioningSources_CommissioningTask_H

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "boost/cstdint.hpp"

class DaqMonitorBEInterface;
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
  
  CommissioningTask( DaqMonitorBEInterface*, const FedChannelConnection& );
  virtual ~CommissioningTask();
  
  void bookHistograms();
  void fillHistograms( const SiStripEventSummary&, const edm::DetSet<SiStripRawDigi>& );

  void updateHistograms();
  void updateFreq( int freq ) { updateFreq_ = freq; }
  
 protected:

  string title( string variable, string contents, uint32_t lld_channel );
  
  DaqMonitorBEInterface* dqm_;
  uint32_t updateFreq_;
  uint32_t fillCntr_;
  FedChannelConnection connection_;
  bool booked_;
  
 private:
  
  CommissioningTask() {;}
  
  virtual void book( const FedChannelConnection& );
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& ) = 0;
  virtual void update() = 0;
  
};

#endif // DQM_SiStripCommissioningSources_CommissioningTask_H

