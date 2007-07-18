#ifndef DTCreateSummaryHistos_H
#define DTCreateSummaryHistos_H


/** \class DTCreateSummaryHistos
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/05/22 07:03:05 $
 *  $Revision: 1.3 $
 *  \author  G. Mila - INFN Torino
 *   
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DataFormats/Common/interface/Handle.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "TPostScript.h"

class DTGeometry;

class DTCreateSummaryHistos: public edm::EDAnalyzer{

public:

  /// Constructor
  DTCreateSummaryHistos(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTCreateSummaryHistos();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();


 private:

  int nevents;
  std::string MainFolder;

  edm::ParameterSet parameters;
  edm::ESHandle<DTGeometry> muonGeom;

  // The file which contain the occupancy plot and the digi event plot
  TFile *theFile;
  
  // The *.ps file which contains the summary histos
  TPostScript *psFile;
  std::string PsFileName;
};

#endif
  
  
