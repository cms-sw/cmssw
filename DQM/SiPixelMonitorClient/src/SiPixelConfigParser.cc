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
  boost::property_tree::ptree xml;
  boost::property_tree::read_xml(filename, xml);

  auto it = xml.find("MonElementConfiguration");
  if (it == xml.not_found()) {
    throw cms::Exception("SiPixelConfigParser")
        << "SiPixelConfigParser XML needs to have a MonElementConfiguration node.";
  }
  this->config_ = it->second;
}

static bool readMEListHelper(boost::property_tree::ptree &config, string const &tagname, vector<string> &me_names) {
  for (auto &kv : config) {
    if (kv.first == tagname) {
      for (auto &mekv : kv.second) {
        if (mekv.first == "MonElement") {
          me_names.push_back(mekv.second.get<std::string>("<xmlattr>.name"));
        }
      }
      return true;
    }
  }
  return false;
}

//
// -- Read ME list for the TrackerMap
//
bool SiPixelConfigParser::getMENamesForTrackerMap(string &tkmap_name, vector<string> &me_names) {
  tkmap_name = config_.get<std::string>("TkMap.<xmlattr>.name", "");
  return readMEListHelper(config_, "TkMap", me_names);
}
//
// -- Read Update Frequency for the TrackerMap
//
bool SiPixelConfigParser::getFrequencyForTrackerMap(int &u_freq) {
  u_freq = config_.get<int>("TkMap.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}
//
// -- Get List of MEs for the module tree plots:
//
bool SiPixelConfigParser::getMENamesForTree(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureLevel.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureLevel", me_names);
}
//
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getMENamesForBarrelSummary(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureBarrelLevel.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureBarrelLevel", me_names);
}
bool SiPixelConfigParser::getMENamesForEndcapSummary(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureEndcapLevel.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureEndcapLevel", me_names);
}

bool SiPixelConfigParser::getMENamesForFEDErrorSummary(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureNonDetId.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureNonDetId", me_names);
}
////
// -- Get List of MEs for the summary plot and the
//
bool SiPixelConfigParser::getFrequencyForBarrelSummary(int &u_freq) {
  u_freq = config_.get<int>("SummaryPlot.SubStructureBarrelLevel.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getFrequencyForEndcapSummary(int &u_freq) {
  u_freq = config_.get<int>("SummaryPlot.SubStructureEndcapLevel.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getMENamesForGrandBarrelSummary(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureGrandBarrelLevel.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureGrandBarrelLevel", me_names);
}

bool SiPixelConfigParser::getMENamesForGrandEndcapSummary(string &structure_name, vector<string> &me_names) {
  structure_name = config_.get<std::string>("SummaryPlot.SubStructureGrandEndcapLevel.<xmlattr>.name", "");
  auto it = config_.find("SummaryPlot");
  if (it == config_.not_found())
    return false;
  return readMEListHelper(it->second, "SubStructureGrandEndcapLevel", me_names);
}

bool SiPixelConfigParser::getFrequencyForGrandBarrelSummary(int &u_freq) {
  u_freq = config_.get<int>("SummaryPlot.SubStructureGrandBarrelLevel.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getFrequencyForGrandEndcapSummary(int &u_freq) {
  u_freq = config_.get<int>("SummaryPlot.SubStructureGrandEndcapLevel.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getMessageLimitForQTests(int &u_freq) {
  u_freq = config_.get<int>("QTests.QTestMessageLimit.<xmlattr>.value", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getSourceType(int &u_freq) {
  u_freq = config_.get<int>("Source.SourceType.<xmlattr>.code", -1);
  if (u_freq >= 0)
    return true;
  return false;
}

bool SiPixelConfigParser::getCalibType(int &u_freq) {
  u_freq = config_.get<int>("Calib.CalibType.<xmlattr>.value", -1);
  if (u_freq >= 0)
    return true;
  return false;
}
