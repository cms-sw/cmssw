#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//
// -- Constructor
//
SiStripConfigParser::SiStripConfigParser() {
  edm::LogInfo("SiStripConfigParser") << " Creating SiStripConfigParser "
                                      << "\n";
}

void SiStripConfigParser::getDocument(std::string filename) {
  boost::property_tree::ptree xml;
  boost::property_tree::read_xml(filename, xml);

  auto it = xml.find("MonElementConfiguration");
  if (it == xml.not_found()) {
    throw cms::Exception("SiPixelConfigParser")
        << "SiPixelConfigParser XML needs to have a MonElementConfiguration node.";
  }
  this->config_ = it->second;
}

// -- Get List of MEs for the summary plot and the
//
bool SiStripConfigParser::getMENamesForSummary(std::map<std::string, std::string>& me_names) {
  for (auto& kv : config_) {
    if (kv.first == "SummaryPlot") {
      for (auto& mekv : kv.second) {
        if (mekv.first == "MonElement") {
          auto name = mekv.second.get<std::string>("<xmlattr>.name");
          auto type = mekv.second.get<std::string>("<xmlattr>.type");
          me_names[name] = type;
        }
      }
      return true;
    }
  }
  return false;
}

//
// -- Get List of MEs for the summary plot and the
//
bool SiStripConfigParser::getFrequencyForSummary(int& u_freq) {
  u_freq = config_.get<int>("SummaryPlot.<xmlattr>.update_frequency", -1);
  if (u_freq >= 0)
    return true;
  return false;
}
