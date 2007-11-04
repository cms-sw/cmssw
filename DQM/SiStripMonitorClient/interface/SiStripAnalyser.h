#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  $Date: 2007/10/24 17:13:23 $
 *  $Revision: 1.11 $
 *  \author  S. Dutta INFN-Pisa
 *   
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class DaqMonitorBEInterface;
class SiStripWebInterface;
class SiStripFedCabling;
class SiStripTrackerMapCreator;
 
class SiStripAnalyser: public edm::EDAnalyzer, public evf::ModuleWeb{

public:

  /// Constructor
  SiStripAnalyser(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~SiStripAnalyser();

  void defaultWebPage(xgi::Input *in, xgi::Output *out); 
  void publish(xdata::InfoSpace *){};
  //  void handleWebRequest(xgi::Input *in, xgi::Output *out); 

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& eSetup);

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& eSetup);

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& eSetup);

  /// Begin Luminosity Block
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) ;

  /// End Luminosity Block
  
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Endjob
  void endJob();


  /// Save histograms to a root file

  void saveAll(int irun, int ilumi);

private:

  void createFedTrackerMap();

  DaqMonitorBEInterface* dbe_;
  MonitorUserInterface* mui_;

  SiStripWebInterface* sistripWebInterface_;

  int fileSaveFrequency_;
  int summaryFrequency_;
  int tkMapFrequency_;
  int staticUpdateFrequency_;

  std::string outputFilePath_;

  edm::ESHandle< SiStripFedCabling > fedCabling_;
  SiStripTrackerMapCreator* trackerMapCreator_;
  bool defaultPageCreated_;

  int nLumiSecs_;
};


#endif
