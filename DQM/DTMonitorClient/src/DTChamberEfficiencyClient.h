#ifndef DTChamberEfficiencyClient_H
#define DTChamberEfficiencyClient_H


/** \class DTChamberEfficiencyClient
 * *
 *  DQM Test Client
 *
 *  $Date: 2011/10/31 17:22:23 $
 *  $Revision: 1.7 $
 *  \author  M. Pelliccioni - INFN Torino
 *   
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTChamberId;
class DTLayerId;

class DTChamberEfficiencyClient: public edm::EDAnalyzer{

public:

  /// Constructor
  DTChamberEfficiencyClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTChamberEfficiencyClient();

protected:

  void beginJob();
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endJob();

  /// book the report summary
  void bookHistos();
  
  /// DQM Client Diagnostic
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context);
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void endRun(edm::Run const& run, edm::EventSetup const& c);

private:

  int nevents;
  unsigned int nLumiSegs;
  int prescaleFactor;

  DQMStore* dbe;

  edm::ESHandle<DTGeometry> muonGeom;

  //an histogram of efficiency for each wheel, for each quality scenario
  MonitorElement* summaryHistos[5][2];
  MonitorElement* globalEffSummary;
 
  MonitorElement* globalEffDistr;
  std::map<int, MonitorElement*> EffDistrPerWh;

};

#endif
