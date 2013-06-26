/*
 * \class DQMLumiMonitor
 *
 * DQM Luminosity Monitoring 
 *
 * $Date: 2012/04/20 06:49:07 $
 * $Revision: 1.3 $
 * \author  S. Dutta
 *
*/

#ifndef DQMLUMIMONITOR_H
#define DQMLUMIMONITORH

#include <string>
#include <vector>
#include <map>

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

class DQMLumiMonitor : public edm::EDAnalyzer {
public:
  DQMLumiMonitor( const edm::ParameterSet& );
  ~DQMLumiMonitor();

protected:

  void beginJob();
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup);
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);
  void endRun(edm::Run const& iRun,  edm::EventSetup const& iSetup);
  void endJob();

private:

  void bookHistograms();

  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  edm::InputTag pixelClusterInputTag_;
  std::string lumiRecordName_;
  
  DQMStore * dbe_;
 
  MonitorElement* nClusME_;
  MonitorElement* intLumiVsLSME_;
  MonitorElement* nClusVsLSME_;
  MonitorElement* corrIntLumiAndClusVsLSME_; 

  float intLumi_;
  int nLumi_;
  unsigned long long m_cacheID_;
};

#endif
