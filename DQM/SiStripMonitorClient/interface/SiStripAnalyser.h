#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  $Date: 2007/07/09 20:21:21 $
 *  $Revision: 1.1 $
 *  \author  S. Dutta INFN-Pisa
 *   
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class DaqMonitorBEInterface;
class SiStripWebInterface;
 
class SiStripAnalyser: public edm::EDAnalyzer, public evf::ModuleWeb{

public:

  /// Constructor
  SiStripAnalyser(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripAnalyser();
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  void defaultWebPage(xgi::Input *in, xgi::Output *out); 
  void publish(xdata::InfoSpace *){};
  void handleWebRequest(xgi::Input *in, xgi::Output *out); 

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::EventSetup& c);


  /// Endjob
  void endJob();

  /// Save histograms to a root file

  void saveAll();

private:

  int nevents;

  DaqMonitorBEInterface* dbe;
  MonitorUserInterface* mui_;

  edm::ParameterSet parameters;
  SiStripWebInterface* sistripWebInterface_;

  int tkMapFrequency_;
  int summaryFrequency_;
  int fileSaveFrequency_;
  unsigned int collationFlag_;
  unsigned int runNumber_;
};


#endif
