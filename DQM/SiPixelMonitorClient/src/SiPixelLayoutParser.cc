#include "DQM/SiPixelMonitorClient/interface/SiPixelLayoutParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cassert>

using namespace std;

//
// -- Constructor
//
SiPixelLayoutParser::SiPixelLayoutParser() {
  edm::LogInfo("SiPixelLayoutParser") << " Creating SiPixelLayoutParser "
                                      << "\n";
  cout << " Creating SiPixelLayoutParser " << endl;
}

//
// -- Get list of Layouts for ME groups
//
bool SiPixelLayoutParser::getAllLayouts(map<string, vector<string>> &layouts) {
  // TODO: implement parser based on property_tree.
  assert(!"No longer implemented.");
  return false;
}
