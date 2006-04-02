#ifndef DQM_SiStripCommissioningSources_CommissioningSource_H
#define DQM_SiStripCommissioningSources_CommissioningSource_H

#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
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

  /** Private default constructor. */
  CommissioningSource();

  bool createTask( const edm::EventSetup& setup, SiStripEventSummary::Task task = SiStripEventSummary::UNKNOWN_TASK );

 private: // ----- data members -----

  string inputModuleLabel_;
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

