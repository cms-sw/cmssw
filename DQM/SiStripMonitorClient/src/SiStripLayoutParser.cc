#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cassert>

//
// -- Constructor
//
SiStripLayoutParser::SiStripLayoutParser() {
  edm::LogInfo("SiStripLayoutParser") << " Creating SiStripLayoutParser "
                                      << "\n";
}
void SiStripLayoutParser::getDocument(std::string filepath) {
  // TODO: add new parser based on boost::property_tree.
  assert(!"No longer implemented.");
}
//
// -- Get list of Layouts for ME groups
//
bool SiStripLayoutParser::getAllLayouts(std::map<std::string, std::vector<std::string> >& layouts) { return false; }
