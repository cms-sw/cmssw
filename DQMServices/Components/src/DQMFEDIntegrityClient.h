
/*
 * \class DQMFEDIntegrityClient
 *
 * DQM FED Client
 *
 * $Date: 2010/03/29 18:34:06 $
 * $Revision: 1.8 $
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
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class DQMFEDIntegrityClient : public edm::EDAnalyzer {
public:
  DQMFEDIntegrityClient( const edm::ParameterSet& );
  ~DQMFEDIntegrityClient();

protected:

  void beginJob();
  void beginRun(const edm::Run& r, const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void endLuminosityBlock(const edm::LuminosityBlock& l, const  edm::EventSetup& c);

  void endRun(const edm::Run& r, const edm::EventSetup& c);
  void endJob();

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
