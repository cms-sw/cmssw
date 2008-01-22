#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


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
