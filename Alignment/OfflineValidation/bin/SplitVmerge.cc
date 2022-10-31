#include <cstdlib>
#include <string>
#include <iostream>
#include <numeric>
#include <functional>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>

#include "TString.h"
#include "TASImage.h"

#include "Alignment/OfflineValidation/interface/CompareAlignments.h"
#include "Alignment/OfflineValidation/macros/FitPVResolution.C"
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

  //Set configuration string for CompareAlignments class
  TString filesAndLabels;

  for (const std::pair<std::string, pt::ptree>& childTree : alignments) {
    filesAndLabels += childTree.second.get<std::string>("file") + "/SplitV.root" +
                      childTree.second.get<std::string>("title") + "|" + childTree.second.get<std::string>("color") +
                      "|" + childTree.second.get<std::string>("style") + " , ";
  }

  filesAndLabels.Remove(filesAndLabels.Length() - 3);

  //Do file comparisons
  CompareAlignments comparer(main_tree.get<std::string>("output"));
  comparer.doComparison(filesAndLabels);

  for (const std::pair<std::string, pt::ptree>& childTree : alignments) {
    PVResolution::loadFileList((childTree.second.get<std::string>("file") + "/SplitV.root").c_str(),
			       "PrimaryVertexResolution",
			       childTree.second.get<std::string>("title"),
			       childTree.second.get<int>("color"),
			       childTree.second.get<int>("style"));
  }

  FitPVResolution("","");

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
