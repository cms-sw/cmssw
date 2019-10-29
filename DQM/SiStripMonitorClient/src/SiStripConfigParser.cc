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

void SiStripConfigParser::getDocument(std::string filepath) {
  // TODO: add new parser.
}

// -- Get List of MEs for the summary plot and the
//
bool SiStripConfigParser::getMENamesForSummary(std::map<std::string, std::string>& me_names) { return false; }
//
// -- Get List of MEs for the summary plot and the
//
bool SiStripConfigParser::getFrequencyForSummary(int& u_freq) { return false; }
