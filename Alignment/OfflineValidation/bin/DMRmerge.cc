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

#include "Alignment/OfflineValidation/interface/CompareAlignments.h"
#include "Alignment/OfflineValidation/interface/PlotAlignmentValidation.h"
#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

using namespace std;
using namespace AllInOneConfig;

namespace pt = boost::property_tree;

std::string getVecTokenized(pt::ptree& tree, const std::string& name, const std::string& token) {
  std::string s;

  for (const auto& childTree : tree.get_child(name)) {
    s += childTree.second.get_value<std::string>() + token;
  }

  return s.substr(0, s.size() - token.size());
}

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
  merge_style = global_style.count("DMR") && global_style.get_child("DMR").count("merge")
                    ? global_style.get_child("DMR").get_child("merge")
                    : global_style;

  //Read all configure variables and set default for missing keys
  std::string methods = validation.count("methods") ? getVecTokenized(validation, "methods", ",") : "median,rmsNorm";
  std::string curves = validation.count("curves") ? getVecTokenized(validation, "curves", ",") : "plain";
  std::string rlabel = validation.count("customrighttitle") ? validation.get<std::string>("customrighttitle") : "";
  rlabel = merge_style.count("Rlabel") ? merge_style.get<std::string>("Rlabel") : rlabel;
  std::string cmslabel = merge_style.count("CMSlabel") ? merge_style.get<std::string>("CMSlabel") : "INTERNAL";

  int minimum = validation.count("minimum") ? validation.get<int>("minimum") : 15;

  bool useFit = validation.count("usefit") ? validation.get<bool>("usefit") : false;
  bool bigText = validation.count("bigtext") ? validation.get<bool>("bigtext") : false;

  TkAlStyle::legendheader = validation.count("legendheader") ? validation.get<std::string>("legendheader") : "";
  TkAlStyle::legendoptions =
      validation.count("legendoptions") ? getVecTokenized(validation, "legendoptions", " ") : "mean rms";
  if (TkAlStyle::toStatus(cmslabel) == CUSTOM)
    TkAlStyle::set(CUSTOM, NONE, cmslabel, rlabel);
  else
    TkAlStyle::set(TkAlStyle::toStatus(cmslabel), NONE, "", rlabel);

  std::vector<int> moduleids;
  if (validation.count("moduleid")) {
    for (const auto& childTree : validation.get_child("moduleid")) {
      moduleids.push_back(childTree.second.get_value<int>());
    }
  }

  //Set configuration string for CompareAlignments class in user defined order
  TString filesAndLabels;
  std::vector<std::pair<std::string, pt::ptree>> alignmentsOrdered;
  for (const auto& childTree : alignments) {
    alignmentsOrdered.push_back(childTree);
  }
  std::sort(alignmentsOrdered.begin(),
            alignmentsOrdered.end(),
            [](const std::pair<std::string, pt::ptree>& left, const std::pair<std::string, pt::ptree>& right) {
              return left.second.get<int>("index") < right.second.get<int>("index");
            });
  for (const auto& childTree : alignmentsOrdered) {
    filesAndLabels +=
        childTree.second.get<std::string>("file") + "/DMR.root=" + childTree.second.get<std::string>("title") + "|" +
        childTree.second.get<std::string>("color") + "|" + childTree.second.get<std::string>("style") + " , ";
  }
  filesAndLabels.Remove(filesAndLabels.Length() - 3);

  //Do file comparisons
  CompareAlignments comparer(main_tree.get<std::string>("output"));
  if (TkAlStyle::toStatus(cmslabel) == CUSTOM)
    comparer.doComparison(filesAndLabels, "", cmslabel, rlabel, CUSTOM);
  else
    comparer.doComparison(filesAndLabels, "", "", rlabel, TkAlStyle::toStatus(cmslabel));

  //Create plots in user defined order
  gStyle->SetTitleH(0.07);
  gStyle->SetTitleW(1.00);
  gStyle->SetTitleFont(132);

  PlotAlignmentValidation plotter(bigText);

  for (const auto& childTree : alignmentsOrdered) {
    plotter.loadFileList((childTree.second.get<std::string>("file") + "/DMR.root").c_str(),
                         childTree.second.get<std::string>("title"),
                         childTree.second.get<int>("color"),
                         childTree.second.get<int>("style"));
  }

  plotter.setOutputDir(main_tree.get<std::string>("output"));
  plotter.useFitForDMRplots(useFit);
  plotter.setTreeBaseDir("TrackHitFilter");
  plotter.plotDMR(methods, minimum, curves);
  plotter.plotSurfaceShapes("coarse");
  plotter.plotChi2((main_tree.get<std::string>("output") + "/" + "result.root").c_str());

  for (const int& moduleid : moduleids) {
    plotter.residual_by_moduleID(moduleid);
  }

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
