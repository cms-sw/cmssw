#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelConfigParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;

//
// -- Constructor
//
SiPixelConfigParser::SiPixelConfigParser() {
  edm::LogInfo("SiPixelConfigParser") << " Creating SiPixelConfigParser "
                                      << "\n";
}

void SiPixelConfigParser::getDocument(std::string filename) {
  //TODO: implement new parser using property_tree
}

//
// -- Read ME list for the TrackerMap
//
bool SiPixelConfigParser::getMENamesForTrackerMap(string &tkmap_name, vector<string> &me_names) { return false; }
//
// -- Read Update Frequency for the TrackerMap
//
bool SiPixelConfigParser::getFrequencyForTrackerMap(int &u_freq) { return false; }
//
// -- Get List of MEs for the module tree plots:
//
bool SiPixelConfigParser::getMENamesForTree(string &structure_name, vector<string> &me_names) { return false; }
//
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getMENamesForBarrelSummary(string &structure_name, vector<string> &me_names) { return false; }
bool SiPixelConfigParser::getMENamesForEndcapSummary(string &structure_name, vector<string> &me_names) { return false; }

bool SiPixelConfigParser::getMENamesForFEDErrorSummary(string &structure_name, vector<string> &me_names) {
  return false;
}
////
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getFrequencyForBarrelSummary(int &u_freq) { return false; }

bool SiPixelConfigParser::getFrequencyForEndcapSummary(int &u_freq) { return false; }

bool SiPixelConfigParser::getMENamesForGrandBarrelSummary(string &structure_name, vector<string> &me_names) {
  return false;
}

bool SiPixelConfigParser::getMENamesForGrandEndcapSummary(string &structure_name, vector<string> &me_names) {
  return false;
}

bool SiPixelConfigParser::getFrequencyForGrandBarrelSummary(int &u_freq) { return false; }

bool SiPixelConfigParser::getFrequencyForGrandEndcapSummary(int &u_freq) { return false; }

bool SiPixelConfigParser::getMessageLimitForQTests(int &u_freq) { return false; }

bool SiPixelConfigParser::getSourceType(int &u_freq) { return false; }

bool SiPixelConfigParser::getCalibType(int &u_freq) { return false; }
