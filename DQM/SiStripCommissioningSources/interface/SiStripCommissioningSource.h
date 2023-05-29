#ifndef DQM_SiStripCommissioningSources_SiStripCommissioningSource_H
#define DQM_SiStripCommissioningSources_SiStripCommissioningSource_H

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include <string>
#include <vector>
#include <map>
#include <cstdint>

class CommissioningTask;
class FedChannelConnection;
class SiStripEventSummary;

/**
   @class SiStripCommissioningSource
*/
class SiStripCommissioningSource : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:  // ---------- Public interface ----------
  /** Map of task objects, identified through FedChanelId */
  typedef std::map<unsigned int, CommissioningTask*> TaskMap;
  typedef std::vector<CommissioningTask*> VecOfTasks;
  typedef std::vector<VecOfTasks> VecOfVecOfTasks;
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  SiStripCommissioningSource(const edm::ParameterSet&);

  /** Private default constructor. */
  SiStripCommissioningSource() = delete;

  ~SiStripCommissioningSource() override;

  void beginRun(edm::Run const&, const edm::EventSetup&) override;
  void endRun(edm::Run const&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

private:  // ---------- Private methods ----------
  /** */
  DQMStore* const dqm(std::string method = "") const;

  /** */
  void createRunNumber();

  /** */
  void createTask(const SiStripEventSummary* const, const edm::EventSetup&);

  /** */
  void createCablingTasks();

  /** */
  void createTasks(sistrip::RunType, const edm::EventSetup&);

  /** */
  void clearCablingTasks();

  /** */
  void clearTasks();

  /** */
  void fillCablingHistos(const SiStripEventSummary* const, const edm::DetSetVector<SiStripRawDigi>&);

  /** */
  void fillHistos(const SiStripEventSummary* const,
                  const edm::DetSetVector<SiStripRawDigi>&,
                  const edm::DetSetVector<SiStripRawDigi>& = edm::DetSetVector<SiStripRawDigi>(),
                  const edmNew::DetSetVector<SiStripCluster>& = edmNew::DetSetVector<SiStripCluster>());

  /** */
  void remove();

  /** */
  void directory(std::stringstream&, uint32_t run_number = 0);

  /** */
  //void cablingForConnectionRun( const sistrip::RunType& ); //@@ do not use!

  // ---------- DQM fwk and cabling ----------

  /** Interface to Data Quality Monitoring framework. */
  DQMStore* dqm_;

  /** */
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  SiStripFedCabling* fedCabling_;

  /** */
  SiStripFecCabling* fecCabling_;

  // ---------- Input / output ----------
  edm::EDGetTokenT<SiStripEventSummary> inputModuleSummaryToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > digiVirginRawToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > digiReorderedToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > digiScopeModeToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripRawDigi> > digiFineDelaySelectionToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clustersToken_;

  /** Name of digi input module. */
  std::string inputModuleLabel_;
  std::string inputModuleLabelAlt_;
  std::string inputModuleLabelSummary_;
  std::string inputClusterLabel_;

  /** Filename of output root file containing source histos. */
  std::string filename_;

  /** Run number used for naming of root file. */
  uint32_t run_;

  /** Partition name used for root files in spy */
  std::string partitionName_;

  /** Record of time used to calculate event rate. */
  int32_t time_;

  /** Is this a spy run ? */
  bool isSpy_;

  // ---------- Histogram-related ----------

  /** Identifies commissioning task read from cfg file. */
  std::string taskConfigurable_;

  /** Identifies commissioning task. */
  sistrip::RunType task_;

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

  /** */
  std::string base_;

  /** flag for choosing the organizational 'view' the DQM histogram tree */
  std::string view_;

  /** parameters to pass to the tasks */
  edm::ParameterSet parameters_;

  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
};

#endif  // DQM_SiStripCommissioningSources_SiStripCommissioningSource_H
