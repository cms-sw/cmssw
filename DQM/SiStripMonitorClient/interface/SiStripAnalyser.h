#ifndef SiStripAnalyser_H
#define SiStripAnalyser_H

/** \class SiStripAnalyser
 * *
 *  SiStrip SiStripAnalyser
 *  $Date: 2007/09/19 14:25:52 $
 *  $Revision: 1.8 $
 *  \author  S. Dutta INFN-Pisa
 *   
 */

#include "DQMServices/Components/interface/DQMAnalyzer.h"
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
class TrackerMapCreator;
 
class SiStripAnalyser: public DQMAnalyzer, public evf::ModuleWeb{

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
  void beginRun(const edm::EventSetup& eSetup);

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

  MonitorUserInterface* mui_;

  SiStripWebInterface* sistripWebInterface_;

  int tkMapFrequency_;
  int summaryFrequency_;
  int fileSaveFrequency_;
  unsigned int collationFlag_;
  unsigned int staticUpdateFrequency_;

  std::string outputFilePath_;

  edm::ESHandle< SiStripFedCabling > fedCabling_;
  TrackerMapCreator* trackerMapCreator_;
  bool defaultPageCreated_;
};


#endif
