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

#include "Alignment/OfflineValidation/macros/trackSplitPlot.h"
#include "Alignment/OfflineValidation/macros/trackSplitPlot.C"
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

  int iov = validation.count("IOV") ? validation.get<int>("IOV") : 1;
  std::string rlabel = validation.count("customrighttitle") ? validation.get<std::string>("customrighttitle") : "";
  rlabel = merge_style.count("Rlabel") ? merge_style.get<std::string>("Rlabel") : rlabel;
  std::string cmslabel = merge_style.count("CMSlabel") ? merge_style.get<std::string>("CMSlabel") : "INTERNAL";
  if (TkAlStyle::toStatus(cmslabel) == CUSTOM)
    TkAlStyle::set(CUSTOM, NONE, cmslabel, rlabel);
  else
    TkAlStyle::set(TkAlStyle::toStatus(cmslabel), NONE, "", rlabel);

  TString filesAndLabels;
  for (const auto& childTree : alignments) {
    // Print node name and its attributes
    std::cout << "Node: " << childTree.first << std::endl;
    for (const auto& attr : childTree.second) {
      std::cout << "  Attribute: " << attr.first << " = " << attr.second.data() << std::endl;
    }

    //std::cout << childTree.second.get<string>("file") << std::endl;
    //std::cout << childTree.second.get<string>("title") << std::endl;
    //std::cout << childTree.second.get<int>("color") << std::endl;
    //std::cout << childTree.second.get<int>("style") << std::endl;

    std::string toAdd = childTree.second.get<string>("file") +
                        Form("/MTSValidation_%s_%d.root=", childTree.first.c_str(), iov) +
                        childTree.second.get<string>("title") +
                        Form("|%i|%i,", childTree.second.get<int>("color"), childTree.second.get<int>("style"));
    filesAndLabels += toAdd;
  }

  std::cout << "filesAndLabels: " << filesAndLabels << std::endl;

  TkAlStyle::legendheader = "";
  TkAlStyle::legendoptions = "all";
  outliercut = -1.0;
  //fillmatrix();
  subdetector = "PIXEL";
  makePlots(filesAndLabels, "./");

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
-- dummy change --
