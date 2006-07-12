#ifndef DQM_SiStripCommissioningSources_SiStripCommissioningSource_H
#define DQM_SiStripCommissioningSources_SiStripCommissioningSource_H

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <string>
#include <map>

class DaqMonitorBEInterface;
class CommissioningTask;
class FedChannelConnection;

/**
   @class SiStripCommissioningSource
*/
class SiStripCommissioningSource : public edm::EDAnalyzer {
  
 public: // ----- public interface -----
  
  /** Map of task objects, identified through FedChanelId */
  typedef std::map<unsigned int, CommissioningTask*> TaskMap;
  
  SiStripCommissioningSource( const edm::ParameterSet& );
  ~SiStripCommissioningSource();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();
  
 private: // ----- private methods -----

  /** Private default constructor. */
  SiStripCommissioningSource();
  
  /** */
  void createTask( sistrip::Task task );
  
  /** */
  DaqMonitorBEInterface* const dqm( std::string method = "" ) const;
  
 private: // ----- data members -----

  std::string inputModuleLabel_;
  /** Interface to Data Quality Monitoring framework. */
  DaqMonitorBEInterface* dqm_;
  /** Identifies commissioning task. */
  std::string task_; 
  /** Map of task objects, identified through FedChanKey. */
  TaskMap tasks_;
  /** */
  int updateFreq_;

  /** */
  std::string filename_;

  /** */
  uint32_t run_;
  /** */
  bool firstEvent_;

  /** */
  SiStripFedCabling* fedCabling_;
  /** */
  SiStripFecCabling* fecCabling_;
  /** */
  bool cablingTask_;
  
};

#endif // DQM_SiStripCommissioningSources_SiStripCommissioningSource_H

