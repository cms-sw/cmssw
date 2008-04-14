#ifndef DQM_SiStripCommissioningSources_LatencyTask_h
#define DQM_SiStripCommissioningSources_LatencyTask_h

#include "DQM/SiStripCommissioningSources/interface/CommissioningTask.h"
#include <map>
#include <string>

/**
   @class LatencyTask
*/
class LatencyTask : public CommissioningTask {

 public:
  
  LatencyTask( DQMStore*, const FedChannelConnection& );
  virtual ~LatencyTask();
  
 private:

  virtual void book();
  virtual void fill( const SiStripEventSummary&,
		     const edm::DetSet<SiStripRawDigi>& );
  virtual void update();
  
  static std::map<std::string, HistoSet> timingMap_;
  static std::map<std::string, HistoSet> clusterMap_;
  HistoSet* timing_;
  HistoSet* cluster_;
  int firstReading_;

};

#endif // DQM_SiStripCommissioningSources_LatencyTask_h

