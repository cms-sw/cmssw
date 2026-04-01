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

#include "Alignment/OfflineValidation/interface/CompareAlignments.h"
#include "Alignment/OfflineValidation/macros/DiMuonMassProfiles.C"
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
  pt::ptree global_style;
  pt::ptree merge_style;
  global_style = main_tree.count("style") ? main_tree.get_child("style") : global_style;
  merge_style = global_style.count("Zmumu") && global_style.get_child("Zmumu").count("merge")
                    ? global_style.get_child("Zmumu").get_child("merge")
                    : global_style;

  //Read all configure variables and set default for missing keys
  bool autoLimits = validation.count("autoLimits") ? validation.get<bool>("autoLimits") : false;

  std::string rlabel = validation.count("customrighttitle") ? validation.get<std::string>("customrighttitle") : "";
  rlabel = merge_style.count("Rlabel") ? merge_style.get<std::string>("Rlabel") : rlabel;
  std::string cmslabel = merge_style.count("CMSlabel") ? merge_style.get<std::string>("CMSlabel") : "INTERNAL";
  std::string outdir = main_tree.count("output") ? main_tree.get<std::string>("output") : "";

  if (TkAlStyle::toStatus(cmslabel) == CUSTOM) {
    TkAlStyle::set(CUSTOM, NONE, cmslabel, rlabel);
  } else {
    TkAlStyle::set(TkAlStyle::toStatus(cmslabel), NONE, "", rlabel);
  }

  TString filesAndLabels;
  for (const auto& childTree : alignments) {
    std::string file = childTree.second.get<string>("file");
    std::cout << file << std::endl;
    std::cout << childTree.second.get<string>("title") << std::endl;

    // Check if the file contains "/eos/cms/" and add the prefix accordingly
    std::string prefixToAdd = file.find("/eos/cms/") != std::string::npos ? "root://eoscms.cern.ch/" : "";
    std::string toAdd = prefixToAdd + file + "/Zmumu.root=" + childTree.second.get<string>("title") + ",";
    filesAndLabels += toAdd;
  }

  if (filesAndLabels.Length() > 0) {
    filesAndLabels.Remove(filesAndLabels.Length() - 1);  // Remove the last character
  }

  std::cout << "filesAndLabels: " << filesAndLabels << std::endl;

  //And finally fit
  DiMuonMassProfiles(filesAndLabels, cmslabel, rlabel, autoLimits);

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
