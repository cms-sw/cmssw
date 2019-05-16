
/*
 * \class DQMFEDIntegrityClient
 *
 * DQM FED Client
 *
 * \author  M. Marienfeld
 *
*/

#ifndef DQMFEDINTEGRITYCLIENT_H
#define DQMFEDINTEGRITYCLIENT_H

#include <string>
#include <vector>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

//
// class declaration
//

class DQMFEDIntegrityClient : public edm::EDAnalyzer {
public:
  DQMFEDIntegrityClient( const edm::ParameterSet& );
  ~DQMFEDIntegrityClient() override;

protected:

  void beginJob() override;
  void beginRun(const edm::Run& r, const edm::EventSetup& c) override;

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override ;

  void endLuminosityBlock(const edm::LuminosityBlock& l, const  edm::EventSetup& c) override;

  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endJob() override;

private:

  void initialize();
  void fillHistograms();
   
  edm::ParameterSet parameters_;

  DQMStore * dbe_;

  // ---------- member data ----------

  int   NBINS;
  float XMIN, XMAX;
  float SummaryContent[10];

  MonitorElement * FedEntries;
  MonitorElement * FedFatal;
  MonitorElement * FedNonFatal;

  MonitorElement * reportSummary;
  MonitorElement * reportSummaryContent[10];
  MonitorElement * reportSummaryMap;

  bool fillInEventloop;
  bool fillOnEndRun;
  bool fillOnEndJob;
  bool fillOnEndLumi;
  std::string moduleName;
  std::string fedFolderName;

};

#endif
