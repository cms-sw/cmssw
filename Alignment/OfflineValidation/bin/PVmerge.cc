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
#include "Alignment/OfflineValidation/macros/FitPVResiduals.C"
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
  merge_style = global_style.count("PV") && global_style.get_child("PV").count("merge")
                    ? global_style.get_child("PV").get_child("merge")
                    : global_style;

  //Read all configure variables and set default for missing keys
  bool doMaps = validation.count("doMaps") ? validation.get<bool>("doMaps") : false;
  bool stdResiduals = validation.count("stdResiduals") ? validation.get<bool>("stdResiduals") : true;
  bool autoLimits = validation.count("autoLimits") ? validation.get<bool>("autoLimits") : false;

  float m_dxyPhiMax = validation.count("m_dxyPhiMax") ? validation.get<float>("m_dxyPhiMax") : 40;
  float m_dzPhiMax = validation.count("m_dzPhiMax") ? validation.get<float>("m_dzPhiMax") : 40;
  float m_dxyEtaMax = validation.count("m_dxyEtaMax") ? validation.get<float>("m_dxyEtaMax") : 40;
  float m_dzEtaMax = validation.count("m_dzEtaMax") ? validation.get<float>("m_dzEtaMax") : 40;
  float m_dxyPtMax = validation.count("m_dxyPtMax") ? validation.get<float>("m_dxyPtMax") : 40;
  float m_dzPtMax = validation.count("m_dzPtMax") ? validation.get<float>("m_dzPtMax") : 40;
  float m_dxyPhiNormMax = validation.count("m_dxyPhiNormMax") ? validation.get<float>("m_dxyPhiNormMax") : 0.5;
  float m_dzPhiNormMax = validation.count("m_dzPhiNormMax") ? validation.get<float>("m_dzPhiNormMax") : 0.5;
  float m_dxyEtaNormMax = validation.count("m_dxyEtaNormMax") ? validation.get<float>("m_dxyEtaNormMax") : 0.5;
  float m_dzEtaNormMax = validation.count("m_dzEtaNormMax") ? validation.get<float>("m_dzEtaNormMax") : 0.5;
  float m_dxyPtNormMax = validation.count("m_dxyPtNormMax") ? validation.get<float>("m_dxyPtNormMax") : 0.5;
  float m_dzPtNormMax = validation.count("m_dzPtNormMax") ? validation.get<float>("m_dzPtNormMax") : 0.5;
  float w_dxyPhiMax = validation.count("w_dxyPhiMax") ? validation.get<float>("w_dxyPhiMax") : 150;
  float w_dzPhiMax = validation.count("w_dzPhiMax") ? validation.get<float>("w_dzPhiMax") : 150;
  float w_dxyEtaMax = validation.count("w_dxyEtaMax") ? validation.get<float>("w_dxyEtaMax") : 150;
  float w_dzEtaMax = validation.count("w_dzEtaMax") ? validation.get<float>("w_dzEtaMax") : 1000;
  float w_dxyPtMax = validation.count("w_dxyPtMax") ? validation.get<float>("w_dxyPtMax") : 150;
  float w_dzPtMax = validation.count("w_dzPtMax") ? validation.get<float>("w_dzPtMax") : 150;
  float w_dxyPhiNormMax = validation.count("w_dxyPhiNormMax") ? validation.get<float>("w_dxyPhiNormMax") : 1.8;
  float w_dzPhiNormMax = validation.count("w_dzPhiNormMax") ? validation.get<float>("w_dzPhiNormMax") : 1.8;
  float w_dxyEtaNormMax = validation.count("w_dxyEtaNormMax") ? validation.get<float>("w_dxyEtaNormMax") : 1.8;
  float w_dzEtaNormMax = validation.count("w_dzEtaNormMax") ? validation.get<float>("w_dzEtaNormMax") : 1.8;
  float w_dxyPtNormMax = validation.count("w_dxyPtNormMax") ? validation.get<float>("w_dxyPtNormMax") : 1.8;
  float w_dzPtNormMax = validation.count("w_dzPtNormMax") ? validation.get<float>("w_dzPtNormMax") : 1.8;
  int iov = validation.count("IOV") ? validation.get<int>("IOV") : 1;
  std::string rlabel = validation.count("customrighttitle") ? validation.get<std::string>("customrighttitle") : "";
  rlabel = merge_style.count("Rlabel") ? merge_style.get<std::string>("Rlabel") : rlabel;
  std::string cmslabel = merge_style.count("CMSlabel") ? merge_style.get<std::string>("CMSlabel") : "INTERNAL";
  if (TkAlStyle::toStatus(cmslabel) == CUSTOM)
    TkAlStyle::set(CUSTOM, NONE, cmslabel, rlabel);
  else
    TkAlStyle::set(TkAlStyle::toStatus(cmslabel), NONE, "", rlabel);

  //Create plots
  // initialize the plot y-axis ranges
  thePlotLimits->init(m_dxyPhiMax,      // mean of dxy vs Phi
                      m_dzPhiMax,       // mean of dz  vs Phi
                      m_dxyEtaMax,      // mean of dxy vs Eta
                      m_dzEtaMax,       // mean of dz  vs Eta
                      m_dxyPtMax,       // mean of dxy vs Pt
                      m_dzPtMax,        // mean of dz  vs Pt
                      m_dxyPhiNormMax,  // mean of dxy vs Phi (norm)
                      m_dzPhiNormMax,   // mean of dz  vs Phi (norm)
                      m_dxyEtaNormMax,  // mean of dxy vs Eta (norm)
                      m_dzEtaNormMax,   // mean of dz  vs Eta (norm)
                      m_dxyPtNormMax,   // mean of dxy vs Pt  (norm)
                      m_dzPtNormMax,    // mean of dz  vs Pt  (norm)
                      w_dxyPhiMax,      // width of dxy vs Phi
                      w_dzPhiMax,       // width of dz  vs Phi
                      w_dxyEtaMax,      // width of dxy vs Eta
                      w_dzEtaMax,       // width of dz  vs Eta
                      w_dxyPtMax,       // width of dxy vs Pt
                      w_dzPtMax,        // width of dz  vs Pt
                      w_dxyPhiNormMax,  // width of dxy vs Phi (norm)
                      w_dzPhiNormMax,   // width of dz  vs Phi (norm)
                      w_dxyEtaNormMax,  // width of dxy vs Eta (norm)
                      w_dzEtaNormMax,   // width of dz  vs Eta (norm)
                      w_dxyPtNormMax,   // width of dxy vs Pt  (norm)
                      w_dzPtNormMax     // width of dz  vs Pt  (norm)
  );

  //Load file list in user defined order
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
    if (childTree.second.get<bool>("isMC")) {
      loadFileList(
          (childTree.second.get<string>("file") + Form("/PVValidation_%s_%d.root", childTree.first.c_str(), 1)).c_str(),
          "PVValidation",
          childTree.second.get<string>("title"),
          childTree.second.get<int>("color"),
          childTree.second.get<int>("style"));
    } else {
      loadFileList(
          (childTree.second.get<string>("file") + Form("/PVValidation_%s_%d.root", childTree.first.c_str(), iov))
              .c_str(),
          "PVValidation",
          childTree.second.get<string>("title"),
          childTree.second.get<int>("color"),
          childTree.second.get<int>("style"));
    }
  }

  //And finally fit
  FitPVResiduals("", stdResiduals, doMaps, "", autoLimits, cmslabel, rlabel);

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<merge>(argc, argv); }
#endif
