
#ifndef DQM_SiStripCommissioningClients_SiStripCommissioningOfflineClient_H
#define DQM_SiStripCommissioningClients_SiStripCommissioningOfflineClient_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripTFile.h"
#include "DQM/SiStripCommissioningClients/interface/SummaryPlotXmlParser.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryPlot.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>
#include <vector>
#include <map>

class CommissioningHistograms;
class TH1;

/**
   @class SiStripCommissioningOfflineClient 
   @author M.Wingham, R.Bainbridge
   
   @brief Class which reads a root file containing "commissioning
   histograms", analyzes the histograms to extract "monitorables", and
   creates summary histograms.
*/
class SiStripCommissioningOfflineClient : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  typedef dqm::harvesting::MonitorElement MonitorElement;
  typedef dqm::harvesting::DQMStore DQMStore;

  SiStripCommissioningOfflineClient(const edm::ParameterSet&);
  ~SiStripCommissioningOfflineClient() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

protected:
  virtual void createHistos(const edm::ParameterSet&, const edm::EventSetup&);
  virtual void uploadToConfigDb() { ; }
  virtual void setInputFiles(std::vector<std::string>&, const std::string, const std::string, uint32_t, bool);

protected:
  /** DQMStore object. */
  DQMStore* bei_;

  /** Action "executor" */
  CommissioningHistograms* histos_;

  /** Input .root file. */
  std::vector<std::string> inputFiles_;

  /** Output .root file. */
  std::string outputFileName_;

  /** */
  bool collateHistos_;

  /** */
  bool analyzeHistos_;

  /** Input .xml file. */
  std::string xmlFile_;

  /** Flag. */
  bool createSummaryPlots_;

  /** */
  bool clientHistos_;

  /** */
  bool uploadToDb_;

  /** Commissioning runType. */
  sistrip::RunType runType_;

  /** Run number. */
  uint32_t runNumber_;

  /** Partition Name */
  std::string partitionName_;

  /** */
  typedef std::vector<TH1*> Histos;

  /** */
  typedef std::map<uint32_t, Histos> HistosMap;

  /** Map containing commissioning histograms. */
  HistosMap map_;

  /** SummaryPlot objects. */
  std::vector<SummaryPlot> plots_;

  /** */
  edm::ParameterSet parameters_;
};

#endif  // DQM_SiStripCommissioningClients_SiStripCommissioningOfflineClient_H
