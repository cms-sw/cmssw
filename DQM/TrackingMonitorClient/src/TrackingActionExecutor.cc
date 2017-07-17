
#include "DQM/TrackingMonitorClient/interface/TrackingActionExecutor.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQM/TrackingMonitorClient/interface/TrackingQualityChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iomanip>
//
// -- Constructor
// 
TrackingActionExecutor::TrackingActionExecutor(edm::ParameterSet const& ps):pSet_(ps) {
  edm::LogInfo("TrackingActionExecutor") << " Creating TrackingActionExecutor " << "\n" ;
  qualityChecker_ = NULL; 
  configWriter_   = NULL;
}
//
// --  Destructor
// 
TrackingActionExecutor::~TrackingActionExecutor() {
  //  std::cout << "[TrackingActionExecutor::~TrackingActionExecutor] .. starting" << std::endl;
  edm::LogInfo("TrackingActionExecutor") << " Deleting TrackingActionExecutor " << "\n" ;
  if (qualityChecker_) delete qualityChecker_;
}

//
// -- Create Status Monitor Elements
//
void TrackingActionExecutor::createGlobalStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){
  if (!qualityChecker_) qualityChecker_ = new TrackingQualityChecker(pSet_);
  qualityChecker_->bookGlobalStatus(ibooker,igetter);
}

void TrackingActionExecutor::createLSStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){
  if (!qualityChecker_) qualityChecker_ = new TrackingQualityChecker(pSet_);
  qualityChecker_->bookLSStatus(ibooker,igetter);
}

//
// -- Fill Dummy Status
//
void TrackingActionExecutor::fillDummyGlobalStatus(){
  qualityChecker_->fillDummyGlobalStatus();
}

void TrackingActionExecutor::fillDummyLSStatus(){
  qualityChecker_->fillDummyLSStatus();
}

//
// -- Fill Status
//
void TrackingActionExecutor::fillGlobalStatus(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  qualityChecker_->fillGlobalStatus(ibooker,igetter);
}
//
// -- Fill Lumi Status
//
void TrackingActionExecutor::fillStatusAtLumi(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
  qualityChecker_->fillLSStatus(ibooker,igetter);
}
//
// -- 
//
void TrackingActionExecutor::createDummyShiftReport(){
  //  std::cout << "[TrackingActionExecutor::createDummyShiftReport]" << std::endl;
  std::ofstream report_file;
  report_file.open("tracking_shift_report.txt", std::ios::out);
  report_file << " Nothing to report!!" << std::endl;
  report_file.close();
}
//
// -- Create Shift Report
//
void TrackingActionExecutor::createShiftReport(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){

  //  std::cout << "[TrackingActionExecutor::createShiftReport]" << std::endl;

  // Read layout configuration
  std::string localPath = std::string("DQM/TrackingMonitorClient/data/tracking_plot_layout.xml");
  SiStripLayoutParser layout_parser;
  layout_parser.getDocument(edm::FileInPath(localPath).fullPath());
    
  std::map<std::string, std::vector<std::string> > layout_map;
  if (!layout_parser.getAllLayouts(layout_map)) return;

  
  std::ostringstream shift_summary;
  if (configWriter_) delete configWriter_;
  configWriter_ = new SiStripConfigWriter();
  configWriter_->init("ShiftReport");


  // Print Report Summary Content
  shift_summary << " Report Summary Content :" << std::endl;  
  shift_summary << " =========================" << std::endl;  
  configWriter_->createElement("ReportSummary");
  
  shift_summary << std::endl;
  printShiftHistoParameters(ibooker,igetter, layout_map, shift_summary);
  
  std::ofstream report_file;
  report_file.open("tracking_shift_report.txt", std::ios::out);
  report_file << shift_summary.str() << std::endl;
  report_file.close();
  configWriter_->write("tracking_shift_report.xml");
  delete configWriter_;
  configWriter_ = 0;
}
//
//  -- Print Report Summary
//
void TrackingActionExecutor::printReportSummary(MonitorElement* me,
					       std::ostringstream& str_val, std::string name) { 

  //  std::cout << "[TrackingActionExecutor::printReportSummary]" << std::endl;
  str_val <<" " << name << "  : ";
  std::string value;
  SiStripUtility::getMEValue(me, value);
  configWriter_->createChildElement("MonitorElement", name, "value", value);
  float fvalue = atof(value.c_str());
  if (fvalue == -1.0)  str_val <<" Dummy Value "<<std::endl;
  else                 str_val << fvalue << std::endl;
}
//
//  -- Print Shift Histogram Properties
//
void TrackingActionExecutor::printShiftHistoParameters(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, std::map<std::string, std::vector<std::string> >& layout_map, std::ostringstream& str_val) { 

  //  std::cout << "[TrackingActionExecutor::printShiftHistoParameters]" << std::endl;
  str_val << std::endl;
  for (std::map<std::string, std::vector< std::string > >::iterator it = layout_map.begin() ; it != layout_map.end(); it++) {
    std::string set_name = it->first;
    if (set_name.find("Summary") != std::string::npos) continue;
    configWriter_->createElement(set_name);
    
    str_val << " " << set_name << " : " << std::endl;
    str_val << " ===================================="<< std::endl;
    
    str_val << std::setprecision(2);
    str_val << setiosflags(std::ios::fixed);
    for (std::vector<std::string>::iterator im = it->second.begin(); 
	 im != it->second.end(); im++) {  
      std::string path_name = (*im);
      if (path_name.size() == 0) continue;
      MonitorElement* me = igetter.get(path_name);
      std::ostringstream entry_str, mean_str, rms_str;
      entry_str << std::setprecision(2);
      entry_str << setiosflags(std::ios::fixed);
      mean_str << std::setprecision(2);
      mean_str << setiosflags(std::ios::fixed);
      rms_str << std::setprecision(2);
      rms_str << setiosflags(std::ios::fixed);
      entry_str << std::setw(7) << me->getEntries();
      mean_str << std::setw(7) << me->getMean();
      rms_str << std::setw(7) << me->getRMS();
      configWriter_->createChildElement("MonitorElement", me->getName(), 
	"entries",entry_str.str(),"mean",mean_str.str(),"rms",rms_str.str());
      
      if (me) str_val << " "<< me->getName()  <<" : entries = "<< std::setw(7) 
		      << me->getEntries() << " mean = "<< me->getMean()
		      <<" : rms = "<< me->getRMS()<< std::endl;
    }
    str_val << std::endl;
  }    
}
