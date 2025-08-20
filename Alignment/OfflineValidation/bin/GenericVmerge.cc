#include <cstdlib>
#include <string>
#include <iostream>
#include <numeric>
#include <functional>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include "boost/filesystem.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/optional.hpp"

#include "TString.h"
#include "TASImage.h"
#include "TGraph.h"

#include "Alignment/OfflineValidation/macros/loopAndPlot.C"
#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;

int merge(int argc, char* argv[]) {
  // parse the command line

  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  //Read in AllInOne json config
  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);

  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");

  TString filesAndLabels;
  for (const auto& childTree : alignments) {
    // Print node name and its attributes
    // std::cout << "Node: " << childTree.first << std::endl;
    // for (const auto& attr : childTree.second) {
    //   std::cout << "  Attribute: " << attr.first << " = " << attr.second.data() << std::endl;
    // }

    std::string file = childTree.second.get<string>("file");
    std::cout << file << std::endl;
    std::cout << childTree.second.get<string>("title") << std::endl;

    // Check if the file contains "/eos/cms/" and add the prefix accordingly
    std::string prefixToAdd = file.find("/eos/cms/") != std::string::npos ? "root://eoscms.cern.ch/" : "";
    std::string toAdd = prefixToAdd + file + "/GenericValidation.root=" + childTree.second.get<string>("title") + ",";
    filesAndLabels += toAdd;
  }

  if (filesAndLabels.Length() > 0) {
    filesAndLabels.Remove(filesAndLabels.Length() - 1);  // Remove the last character
  }

  std::cout << "filesAndLabels: " << filesAndLabels << std::endl;

  loopAndPlot(filesAndLabels);

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
