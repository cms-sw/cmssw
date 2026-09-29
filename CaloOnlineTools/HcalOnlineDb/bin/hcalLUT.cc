#include "CaloOnlineTools/HcalOnlineDb/interface/LutXml.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"

#include <iostream>
#include <sstream>
#include <string>

void mergeLUTs(const std::string& flist, const std::string& out) {
  LutXml xmls;
  std::istringstream iss(flist);
  std::string file;
  while (iss >> file) {
    xmls += LutXml(file);
  }
  xmls.write(out);
}

int main(int argc, char** argv) {
  optutl::CommandLineParser parser("hcalLUT");
  parser.parseArguments(argc, argv, true);

  std::string flist = parser.stringValue("storePrepend");
  std::string out = parser.stringValue("outputFile");

  if (flist.empty() or out.empty()) {
    std::cerr << "One or more of arguments \"storePrepend\" and \"outputFile\" is empty !" << std::endl;
    return -1;
  }

  mergeLUTs(flist, out);

  return 0;
}
