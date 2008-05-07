#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  $Date: 2008/04/28 22:39:00 $
 *  $Revision: 1.21 $
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

class DQMStore;
class SiStripWebInterface;
class SiStripFedCabling;
class SiStripDetCabling;
class SiStripActionExecutor;
 
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
  void beginJob(edm::EventSetup const& eSetup);

  /// BeginRun
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Analyze
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  /// Begin Luminosity Block
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) ;

  /// End Luminosity Block
  
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup);

  /// EndRun
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  /// Endjob
  void endJob();



private:


  DQMStore* dqmStore_;

  SiStripWebInterface* sistripWebInterface_;

  int fileSaveFrequency_;
  int summaryFrequency_;
  int tkMapFrequency_;
  int staticUpdateFrequency_;
  bool globalStatusFilling_;

  std::string outputFilePath_;
  std::string outputFileName_;

  edm::ParameterSet tkMapPSet_;
  edm::ESHandle< SiStripFedCabling > fedCabling_;
  edm::ESHandle< SiStripDetCabling > detCabling_;
  SiStripActionExecutor* actionExecutor_;

  unsigned long long m_cacheID_;
  int nLumiSecs_;
  int nEvents_;

  std::ostringstream html_out_;

};


#endif
