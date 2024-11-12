#include <iostream>
#include <stdexcept>
#include <sstream>
#include "TString.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "CaloOnlineTools/HcalOnlineDb/interface/HcalLutManager.h"
#include "FWCore/Utilities/interface/FileInPath.h"

using namespace std;

void mergeLUTs(const char *flist, const char *out) {
  LutXml xmls;
  stringstream ss(flist);
  while (ss.good()) {
    string file;
    ss >> file;
    xmls += LutXml(file);
  }
  xmls.write(out);
}

int main(int argc, char **argv) {
  optutl::CommandLineParser parser("runTestParameters");
  parser.parseArguments(argc, argv, true);
  if (argc < 2) {
    std::cerr << "runTest: missing input command" << std::endl;
  } else if (strcmp(argv[1], "merge") == 0) {
    std::string flist_ = parser.stringValue("storePrepend");
    std::string out_ = parser.stringValue("outputFile");
    mergeLUTs(flist_.c_str(), out_.c_str());
  } else if (strcmp(argv[1], "create-lut-loader") == 0) {
    std::string _file_list = parser.stringValue("outputFile");
    std::string _tag = parser.stringValue("tag");
    std::string _comment = parser.stringValue("storePrepend");
    const std::string &_prefix = _tag;
    std::string _version = "1";
    int _subversion = 0;
    HcalLutManager manager;
    manager.create_lut_loader(_file_list, _prefix, _tag, _comment, _tag, _subversion);
  } else {
    throw std::invalid_argument(Form("Unknown command: %s", argv[1]));
  }

  return 0;
}
