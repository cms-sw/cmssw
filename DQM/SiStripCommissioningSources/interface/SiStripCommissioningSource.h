#ifndef DQM_SiStripCommissioningSources_SiStripCommissioningSource_H
#define DQM_SiStripCommissioningSources_SiStripCommissioningSource_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include <string>
#include <vector>
#include <map>

class DaqMonitorBEInterface;
class CommissioningTask;
class FedChannelConnection;
class SiStripEventSummary;

/**
   @class SiStripCommissioningSource
*/
class SiStripCommissioningSource : public edm::EDAnalyzer {
  
 public: // ----- public interface -----
  
  /** Map of task objects, identified through FedChanelId */
  typedef std::map<unsigned int, CommissioningTask*> TaskMap;
  typedef std::vector<CommissioningTask*> VecOfTasks;
  typedef std::vector<VecOfTasks> VecOfVecOfTasks;
  
  SiStripCommissioningSource( const edm::ParameterSet& );
  ~SiStripCommissioningSource();
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob();
  
 private: // ----- private methods -----

  /** Private default constructor. */
  SiStripCommissioningSource();
  
  /** */
  void createTask( const SiStripEventSummary* const );
  
  /** */
  DaqMonitorBEInterface* const dqm( std::string method = "" ) const;
  
 private: // ----- methods -----

  void createCablingTasks();
  void createTasks();

  void clearCablingTasks();
  void clearTasks();

  void fillCablingHistos( const SiStripEventSummary* const,
			  const edm::DetSetVector<SiStripRawDigi>& );
  void fillHistos( const SiStripEventSummary* const,
		   const edm::DetSetVector<SiStripRawDigi>& );
  
 private:

  // ---------- DQM fwk and cabling ----------

  /** Interface to Data Quality Monitoring framework. */
  DaqMonitorBEInterface* dqm_;

  /** */
  SiStripFedCabling* fedCabling_;

  /** */
  SiStripFecCabling* fecCabling_;
  
  // ---------- Input / output ----------

  /** Name of digi input module. */
  std::string inputModuleLabel_;

  /** Filename of output root file containing source histos. */
  std::string filename_;

  /** Run number used for naming of root file. */
  uint32_t run_;

  /** Record of time used to calculate event rate. */
  uint32_t time_;

  // ---------- Histogram-related ----------

  /** Identifies commissioning task read from cfg file. */
  std::string taskConfigurable_; 

  /** Identifies commissioning task. */
  sistrip::Task task_; 

  /** Vector of vector of task objects (indexed using FED id.ch. */
  VecOfVecOfTasks tasks_;

  /** Map of cabling task objects (indexed using FEC key). */
  TaskMap cablingTasks_;

  /** Flag to indicate whether histo objects exist or not. */
  bool tasksExist_;

  /** Flag to indicate whether task is FED cabling or not. */
  bool cablingTask_;
  
  /** Update frequency for histograms (ignored for cabling). */
  int updateFreq_;

};

#endif // DQM_SiStripCommissioningSources_SiStripCommissioningSource_H

