
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripSummaryCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigWriter.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityChecker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <iomanip>
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
  std::string fpath = pSet_.getUntrackedParameter<std::string>("SummaryConfigPath","DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  if (summaryCreator_->readConfiguration(fpath)) return true;
  else return false;
}
//
// -- Read Configurationn File
//
/*
bool SiStripActionExecutor::readTkMapConfiguration() {
  
  if (tkMapCreator_) delete tkMapCreator_;
  tkMapCreator_ = new SiStripTrackerMapCreator();
  if (tkMapCreator_) return true;
  else return false;
}
*/
bool SiStripActionExecutor::readTkMapConfiguration(const edm::EventSetup& eSetup) {
  
  if (tkMapCreator_) delete tkMapCreator_;
  tkMapCreator_ = new SiStripTrackerMapCreator(eSetup);
  if (tkMapCreator_) return true;
  else return false;
}
//
// -- Create and Fill Summary Monitor Elements
//
void SiStripActionExecutor::createSummary(DQMStore* dqm_store) {
  if (summaryCreator_) {
    dqm_store->cd();
    std::string dname = "SiStrip/MechanicalView";
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
    std::string dname = "MechanicalView";
    if (SiStripUtility::goToDir(dqm_store, dname)) {
      summaryCreator_->createSummary(dqm_store);
    }
    dqm_store->cd();
  }
}
//
// -- create tracker map
//
void SiStripActionExecutor::createTkMap(const edm::ParameterSet & tkmapPset, 
					DQMStore* dqm_store, std::string& map_type,
                                        const edm::EventSetup& eSetup) {
  if (tkMapCreator_) tkMapCreator_->create(tkmapPset, dqm_store, map_type, eSetup);
}
//
// -- create tracker map for offline
//
void SiStripActionExecutor::createOfflineTkMap(const edm::ParameterSet & tkmapPset,
					DQMStore* dqm_store, std::string& map_type,
                                        const edm::EventSetup& eSetup) {
  if (tkMapCreator_) tkMapCreator_->createForOffline(tkmapPset, dqm_store, map_type, eSetup);
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
void SiStripActionExecutor::fillStatus(DQMStore* dqm_store, const edm::ESHandle<SiStripDetCabling>& detcabling, const edm::EventSetup& eSetup) {
  qualityChecker_->fillStatus(dqm_store, detcabling, eSetup);
}
//
// -- Fill Lumi Status
//
void SiStripActionExecutor::fillStatusAtLumi(DQMStore* dqm_store) {
  qualityChecker_->fillStatusAtLumi(dqm_store);
}
//
// -- 
//
void SiStripActionExecutor::createDummyShiftReport(){
  ofstream report_file;
  report_file.open("sistrip_shift_report.txt", std::ios::out);
  report_file << " Nothing to report!!" << std::endl;
  report_file.close();
}
//
// -- Create Shift Report
//
void SiStripActionExecutor::createShiftReport(DQMStore * dqm_store){

  // Read layout configuration
  std::string localPath = std::string("DQM/SiStripMonitorClient/data/sistrip_plot_layout.xml");
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
  
  MonitorElement* me;
  std::string report_path;
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

  shift_summary << std::endl;
  printShiftHistoParameters(dqm_store, layout_map, shift_summary);
  
  ofstream report_file;
  report_file.open("sistrip_shift_report.txt", std::ios::out);
  report_file << shift_summary.str() << std::endl;
  report_file.close();
  configWriter_->write("sistrip_shift_report.xml");
  delete configWriter_;
  configWriter_ = 0;
}
//
//  -- Print Report Summary
//
void SiStripActionExecutor::printReportSummary(MonitorElement* me,
					       std::ostringstream& str_val, std::string name) { 
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
void SiStripActionExecutor::printShiftHistoParameters(DQMStore * dqm_store, std::map<std::string, std::vector<std::string> >& layout_map, std::ostringstream& str_val) { 

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
      MonitorElement* me = dqm_store->get(path_name);
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
//
//  -- Print List of Modules with QTest warning or Error
//
void SiStripActionExecutor::printFaultyModuleList(DQMStore * dqm_store, std::ostringstream& str_val) { 
  dqm_store->cd();

  std::string mdir = "MechanicalView";
  if (!SiStripUtility::goToDir(dqm_store, mdir)) return;
  std::string mechanicalview_dir = dqm_store->pwd();

  std::vector<std::string> subdet_folder;
  subdet_folder.push_back("TIB");
  subdet_folder.push_back("TOB");
  subdet_folder.push_back("TEC/side_1");
  subdet_folder.push_back("TEC/side_2");
  subdet_folder.push_back("TID/side_1");
  subdet_folder.push_back("TID/side_2");
  
  int nDetsTotal = 0;
  int nDetsWithErrorTotal = 0;
  for (std::vector<std::string>::const_iterator im = subdet_folder.begin(); im != subdet_folder.end(); im++) {       
    std::string dname = mechanicalview_dir + "/" + (*im);
    if (!dqm_store->dirExists(dname)) continue;
    str_val << "============"<< std::endl;
    str_val << (*im)         << std::endl;                                                    
    str_val << "============"<< std::endl;
    str_val << std::endl;      

    dqm_store->cd(dname);
    std::vector<std::string> module_folders;
    SiStripUtility::getModuleFolderList(dqm_store, module_folders);
    int nDets = module_folders.size();
    dqm_store->cd();    
  
    int nDetsWithError = 0;
    std::string bad_module_folder = dname + "/" + "BadModuleList";
    if (dqm_store->dirExists(bad_module_folder)) {
      std::vector<MonitorElement *> meVec = dqm_store->getContents(bad_module_folder);
      for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
        nDetsWithError++; 
        uint16_t flag = (*it)->getIntValue();
        std::string message;
	SiStripUtility::getBadModuleStatus(flag, message);
	str_val << (*it)->getName() <<  " flag : " << (*it)->getIntValue() << "  " << message << std::endl;
      } 
    }
    str_val << "--------------------------------------------------------------------"<< std::endl;
    str_val << " Detectors :  Total "<<   nDets
            << " with Error " << nDetsWithError<< std::endl;   
    str_val << "--------------------------------------------------------------------"<< std::endl;
    nDetsTotal += nDets;
    nDetsWithErrorTotal += nDetsWithError;        
  }    
  dqm_store->cd();
  str_val << "--------------------------------------------------------------------"<< std::endl;
  str_val << " Total Number of Connected Detectors : " <<   nDetsTotal << std::endl;
  str_val << " Total Number of Detectors with Error : " << nDetsWithErrorTotal << std::endl;
  str_val << "--------------------------------------------------------------------"<< std::endl;

}
