
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iomanip>
using namespace std;
//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor(edm::ParameterSet const& ps):pSet_(ps) {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
  summaryCreator_ = 0;
  tkMapCreator_   = 0;
  qualityChecker_ = 0; 
  configWriter_   = 0;
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
  if (summaryCreator_) delete summaryCreator_;
  if (tkMapCreator_)   delete tkMapCreator_;
  if (qualityChecker_)  delete qualityChecker_;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration() {
  
  if (!summaryCreator_) {
    summaryCreator_ = new SiStripSummaryCreator();
  }
  string fpath = pSet_.getUntrackedParameter<std::string>("SummaryConfigPath","DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  if (summaryCreator_->readConfiguration(fpath)) return true;
  else return false;
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readTkMapConfiguration() {
  
  if (tkMapCreator_) delete tkMapCreator_;
  tkMapCreator_ = new SiStripTrackerMapCreator();
  if (tkMapCreator_) return true;
  else return false;
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummary(DQMStore* dqm_store) {
  if (summaryCreator_) {
    dqm_store->cd();
    string dname = "SiStrip/MechanicalView";
    if (dqm_store->dirExists(dname)) {
      dqm_store->cd(dname);
      summaryCreator_->createSummary(dqm_store);
    }
  }
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummaryOffline(DQMStore* dqm_store) {
  if (summaryCreator_) {
    dqm_store->cd();
    string dname = "MechanicalView";
    if (SiStripUtility::goToDir(dqm_store, dname)) {
      summaryCreator_->createSummary(dqm_store);
    }
  }
}
//
// -- create tracker map
//
void SiStripActionExecutor::createTkMap(const edm::ParameterSet & tkmapPset, 
       const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store, string& map_type) {
  if (tkMapCreator_) tkMapCreator_->create(tkmapPset, fedcabling, dqm_store, map_type);
}
//
// -- Create Status Monitor Elements
//
void SiStripActionExecutor::createStatus(DQMStore* dqm_store){
  if (!qualityChecker_) qualityChecker_ = new SiStripQualityChecker(pSet_);
  qualityChecker_->bookStatus(dqm_store);
}
//
// -- Fill Dummy Status
//
void SiStripActionExecutor::fillDummyStatus(){
  qualityChecker_->fillDummyStatus();
}
//
// -- Fill Status
//
void SiStripActionExecutor::fillStatus(DQMStore* dqm_store) {
  qualityChecker_->fillStatus(dqm_store);
}
//
// -- 
//
void SiStripActionExecutor::createDummyShiftReport(){
  ofstream report_file;
  report_file.open("sistrip_shift_report.txt", ios::out);
  report_file << " Nothing to report!!" << endl;
  report_file.close();
}
//
// -- Create Shift Report
//
void SiStripActionExecutor::createShiftReport(DQMStore * dqm_store){

  // Read layout configuration
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_plot_layout.xml");
  SiStripLayoutParser* layout_parser = new SiStripLayoutParser();
  layout_parser->getDocument(edm::FileInPath(localPath).fullPath());
    
  map<string, vector<string> > layout_map;
  if (!layout_parser->getAllLayouts(layout_map)) return;
  delete layout_parser;

  
  ostringstream shift_summary;
  if (configWriter_) delete configWriter_;
  configWriter_ = new SiStripConfigWriter();
  configWriter_->init("ShiftReport");


  // Print Report Summary Content
  shift_summary << " Report Summary Content :" << endl;  
  shift_summary << " =========================" << endl;  
  configWriter_->createElement("ReportSummary");
  
  MonitorElement* me;
  string report_path;
  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TECB";
  me  = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TECB"); 

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TECF";
  me = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TECF");
  
  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIB";
  me = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TIB");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIDB";
  me = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TIDB");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TIDF";
  me = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TIDF");

  report_path = "SiStrip/EventInfo/reportSummaryContents/SiStrip_DetFraction_TOB";
  me = dqm_store->get(report_path);    
  printReportSummary(me, shift_summary, "TOB");

  shift_summary << endl;
  printShiftHistoParameters(dqm_store, layout_map, shift_summary);
  
  ofstream report_file;
  report_file.open("sistrip_shift_report.txt", ios::out);
  report_file << shift_summary.str() << endl;
  report_file.close();
  configWriter_->write("sistrip_shift_report.xml");
  delete configWriter_;
  configWriter_ = 0;
}
//
//  -- Print Report Summary
//
void SiStripActionExecutor::printReportSummary(MonitorElement* me,
					       ostringstream& str_val, string name) { 
  str_val <<" " << name << "  : ";
  string value;
  SiStripUtility::getMEValue(me, value);
  configWriter_->createChildElement("MonitorElement", name, "value", value);
  float fvalue = atof(value.c_str());
  if (fvalue == -1.0)  str_val <<" Dummy Value "<<endl;
  else                 str_val << fvalue << endl;
}
//
//  -- Print Shift Histogram Properties
//
void SiStripActionExecutor::printShiftHistoParameters(DQMStore * dqm_store, map<string, vector<string> >& layout_map, ostringstream& str_val) { 

  str_val << endl;
  for (map<std::string, std::vector< std::string > >::iterator it = layout_map.begin() ; it != layout_map.end(); it++) {
    string set_name = it->first;
    if (set_name.find("Summary") != string::npos) continue;
    configWriter_->createElement(set_name);
    
    str_val << " " << set_name << " : " << endl;
    str_val << " ===================================="<< endl;
    
    str_val << setprecision(2);
    str_val << setiosflags(ios::fixed);
    for (vector<string>::iterator im = it->second.begin(); 
	 im != it->second.end(); im++) {  
      string path_name = (*im);
      if (path_name.size() == 0) continue;
      MonitorElement* me = dqm_store->get(path_name);
      ostringstream entry_str, mean_str, rms_str;
      entry_str << setprecision(2);
      entry_str << setiosflags(ios::fixed);
      mean_str << setprecision(2);
      mean_str << setiosflags(ios::fixed);
      rms_str << setprecision(2);
      rms_str << setiosflags(ios::fixed);
      entry_str << setw(7) << me->getEntries();
      mean_str << setw(7) << me->getMean();
      rms_str << setw(7) << me->getRMS();
      configWriter_->createChildElement("MonitorElement", me->getName(), 
	"entries",entry_str.str(),"mean",mean_str.str(),"rms",rms_str.str());
      
      if (me) str_val << " "<< me->getName()  <<" : entries = "<< setw(7) 
		      << me->getEntries() << " mean = "<< me->getMean()
		      <<" : rms = "<< me->getRMS()<< endl;
    }
    str_val << endl;
  }    
}
//
//  -- Print List of Modules with QTest warning or Error
//
void SiStripActionExecutor::printFaultyModuleList(DQMStore * dqm_store, ostringstream& str_val) { 
  qualityChecker_->printFaultyModuleList(dqm_store, str_val);   
}
