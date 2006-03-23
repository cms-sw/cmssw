#ifndef DQM_SiStripCommissioningSources_CommissioningSource_H
#define DQM_SiStripCommissioningSources_CommissioningSource_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <map>

class DaqMonitorBEInterface;
class CommissioningTask;
class FedChannelConnection;

using namespace std;

/**
   @class CommissioningSource
*/
class CommissioningSource : public edm::EDAnalyzer {

 public: // ----- public interface -----
  
  /** May of task objects, identified through FedChanelId */
  typedef map<unsigned int, CommissioningTask*> TaskMap;
  
  CommissioningSource( const edm::ParameterSet& );
  ~CommissioningSource();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();
  
 private: // ----- private methods -----

  CommissioningTask* createTask( const FedChannelConnection& );

 private: // ----- data members -----

  /** Private default constructor. */
  CommissioningSource();

  /** Interface to Data Quality Monitoring framework. */
  DaqMonitorBEInterface* dqm_;
  
  /** Identifies commissioning task. */
  string task_; 
  /** Map of task objects, identified through FedChanKey. */
  TaskMap tasks_;

  /** */
  int updateFreq_;
  
};

#endif // DQM_SiStripCommissioningSources_CommissioningSource_H

