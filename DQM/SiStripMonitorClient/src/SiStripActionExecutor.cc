#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
  //  configParser_ = 0;
  qtHandler_ = 0;
  summaryCreator_= 0;
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
  //  if (configParser_) delete configParser_;
  if (qtHandler_) delete qtHandler_;
  if (summaryCreator_) delete   summaryCreator_;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration() {
  
  if (!summaryCreator_) {
    summaryCreator_ = new SiStripSummaryCreator();
  }
  if (summaryCreator_->readConfiguration()) return true;
  else return false;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration(int& sum_freq) {
  bool result = false;
  if (readConfiguration()) {
    sum_freq = summaryCreator_->getFrequency();
    if (sum_freq != -1) result = true;
  }
  return result;
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummary(DaqMonitorBEInterface* bei) {
  bei->cd();
  summaryCreator_->createSummary(bei);
}
//
// -- Setup Quality Tests 
//
void SiStripActionExecutor::setupQTests(DaqMonitorBEInterface* bei) {
  bei->cd();
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_qualitytest_config.xml");
  if (!qtHandler_) {
    qtHandler_ = new QTestHandle();
  }
  if(!qtHandler_->configureTests(edm::FileInPath(localPath).fullPath(),bei)){
    cout << " Setting Up Quality Tests " << endl;
    qtHandler_->attachTests(bei);			
    bei->cd();
  } else {
    cout << " Problem to Set Up Quality Tests " << endl;
  }
}
//
// -- Save Monitor Elements in a file
//      
void SiStripActionExecutor::saveMEs(DaqMonitorBEInterface* bei, string fname){
   bei->save(fname,bei->pwd(),90);
}
//
// -- Get TkMap ME names
//
//int SiStripActionExecutor::getTkMapMENames(std::vector<std::string>& names) {
//  int nval = tkMapCreator_->getMENames(names);
//  return nval;
//}
