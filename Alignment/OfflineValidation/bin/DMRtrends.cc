#include <cstdlib>
#include <string>
#include <tuple>
#include <iostream>
#include <numeric>
#include <functional>
#include <unistd.h>

#include "TFile.h"
#include "TGraph.h"
#include "TH1.h"

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/optional.hpp"

#include "TString.h"
#include "TColor.h"

#include "Alignment/OfflineValidation/interface/PrepareDMRTrends.h"
#include "Alignment/OfflineValidation/interface/Trend.h"

using namespace std;
using namespace AllInOneConfig;
namespace fs = boost::filesystem;
namespace bc = boost::container;

static const char *bold = "\e[1m", *normal = "\e[0m";
static const float defaultConvertScale = 1000.;
static const int startRun2016 = 272930;
static const int endRun2018 = 325175;

namespace pt = boost::property_tree;

int trends(int argc, char *argv[]) {
  // parse the command line

  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);

  //Read in AllInOne json config
  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);

  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");
  pt::ptree style = main_tree.get_child("style");

  //Read all configure variables and set default for missing keys
  string outputdir = main_tree.get<string>("output");
  bool FORCE = validation.count("FORCE") ? validation.get<bool>("FORCE") : false;
  string year = validation.count("year") ? validation.get<string>("year") : "Run2";
  TString lumiInputFile = style.get_child("trends").count("lumiInputFile")
                              ? style.get_child("trends").get<string>("lumiInputFile")
                              : "Alignment/OfflineValidation/data/lumiPerRun_Run2.txt";
  fs::path lumiFile = lumiInputFile.Data();
  edm::FileInPath fip = edm::FileInPath(lumiFile.string());
  fs::path pathToLumiFile = "";
  if (!fs::exists(lumiFile)) {
    pathToLumiFile = fip.fullPath();
  } else {
    pathToLumiFile = lumiFile;
  }
  if (!fs::exists(pathToLumiFile)) {
    cout << "ERROR: lumi-per-run file (" << lumiFile.string().data() << ") not found!" << endl
         << "Please check!" << endl;
    exit(EXIT_FAILURE);
  } else {
    cout << "Found lumi-per-run file: " << pathToLumiFile.string().data() << endl;
  }
  if (!lumiInputFile.Contains(year)) {
    cout << "ERROR: lumi-per-run file must contain (" << year.data() << ")!" << endl << "Please check!" << endl;
    exit(EXIT_FAILURE);
  }

  string lumiAxisType = "recorded";
  if (lumiInputFile.Contains("delivered"))
    lumiAxisType = "delivered";

  std::cout << Form("NOTE: using %s luminosity!", lumiAxisType.data()) << std::endl;

  vector<int> IOVlist;
  vector<string> inputFiles;
  for (auto const &childTree : validation.get_child("IOV")) {
    int iov = childTree.second.get_value<int>();
    IOVlist.push_back(iov);
    TString mergeFile = validation.get<string>("mergeFile");
    string input = Form("%s/OfflineValidationSummary.root", mergeFile.ReplaceAll("{}", to_string(iov)).Data());
    inputFiles.push_back(input);
  }

  string labels_to_add = "";
  if (validation.count("labels")) {
    for (auto const &label : validation.get_child("labels")) {
      labels_to_add += "_";
      labels_to_add += label.second.get_value<string>();
    }
  }

  fs::path pname = Form("%s/DMRtrends%s.root", outputdir.data(), labels_to_add.data());

  vector<TString> structures{"BPIX", "BPIX_y", "FPIX", "FPIX_y", "TIB", "TID", "TOB", "TEC"};

  map<TString, int> nlayers{{"BPIX", 4}, {"FPIX", 3}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};
  if (year == "2016")
    nlayers = {{"BPIX", 3}, {"FPIX", 2}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};

  PrepareDMRTrends prepareTrends(pname.c_str(), alignments);
  if (validation.count("Variables")) {
    for (auto const &Variable : validation.get_child("Variables")) {
      prepareTrends.compileDMRTrends(
          IOVlist, Variable.second.get_value<string>(), inputFiles, structures, nlayers, FORCE);
    }
  } else
    prepareTrends.compileDMRTrends(IOVlist, "median", inputFiles, structures, nlayers, FORCE);

  assert(fs::exists(pname));

  float convertUnit = style.get_child("trends").count("convertUnit")
                          ? style.get_child("trends").get<float>("convertUnit")
                          : defaultConvertScale;
  int firstRun = validation.count("firstRun") ? validation.get<int>("firstRun") : startRun2016;
  int lastRun = validation.count("lastRun") ? validation.get<int>("lastRun") : endRun2018;

  const Run2Lumi GetLumi(pathToLumiFile.string().data(), firstRun, lastRun, convertUnit);

  auto f = TFile::Open(pname.c_str());

  for (auto const &Variable : validation.get_child("Variables")) {
    vector<tuple<TString, TString, float, float>> DMRs{{"mu", "#mu [#mum]", -6, 6},
                                                       {"sigma", "#sigma_{#mu} [#mum]", -15, 15},
                                                       {"muplus", "#mu outward [#mum]", -6, 6},
                                                       {"sigmaplus", "#sigma_{#mu outward} [#mum]", -15, 15},
                                                       {"muminus", "#mu inward [#mum]", -6, 6},
                                                       {"sigmaminus", "#sigma_{#mu inward} [#mum]", -15, 15},
                                                       {"deltamu", "#Delta#mu [#mum]", -15, 15},
                                                       {"sigmadeltamu", "#sigma_{#Delta#mu} [#mum]", -15, 15},
                                                       {"musigma", "#mu [#mum]", -6, 6},
                                                       {"muplussigmaplus", "#mu outward [#mum]", -15, 15},
                                                       {"muminussigmaminus", "#mu inward [#mum]", -15, 15},
                                                       {"deltamusigmadeltamu", "#Delta#mu [#mum]", -15, 15}};

    if (Variable.second.get_value<string>() == "DrmsNR") {
      DMRs = {{"mu", "RMS(x'_{pred}-x'_{hit} /#sigma)", -1.2, 1.2},
              {"sigma", "#sigma_{RMS(x'_{pred}-x'_{hit} /#sigma)}", -6, 6},
              {"muplus", "RMS(x'_{pred}-x'_{hit} /#sigma) outward", -1.2, 1.2},
              {"sigmaplus", "#sigma_{#mu outward}", -6, 6},
              {"muminus", "RMS(x'_{pred}-x'_{hit} /#sigma) inward", -1.2, 1.2},
              {"sigmaminus", "#sigma_{RMS(x'_{pred}-x'_{hit} /#sigma) inward}", -6, 6},
              {"deltamu", "#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)", -0.15, 0.15},
              {"sigmadeltamu", "#sigma_{#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)}", -6, 6},
              {"musigma", "RMS(x'_{pred}-x'_{hit} /#sigma)", -1.2, 1.2},
              {"muplussigmaplus", "RMS(x'_{pred}-x'_{hit} /#sigma) outward", -1.2, 1.2},
              {"muminussigmaminus", "RMS(x'_{pred}-x'_{hit} /#sigma) inward", -1.2, 1.2},
              {"deltamusigmadeltamu", "#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)", -0.15, 0.15}};
    }

    for (const auto &structure : structures) {
      TString structname = structure;
      structname.ReplaceAll("_y", "");
      size_t layersnumber = nlayers.at(structname);
      for (Size_t layer = 0; layer <= layersnumber; layer++) {
        TString structtitle = "";
        if (structure.Contains("PIX") && !(structure.Contains("_y")))
          structtitle = structure + " (x)";
        else if (structure.Contains("_y")) {
          TString substring(structure(0, 4));
          structtitle = substring + " (y)";
        } else
          structtitle = structure;
        if (layer != 0) {
          if (structure == "TID" || structure == "TEC" || structure == "FPIX" || structure == "FPIX_y")
            structtitle += "  disc ";
          else
            structtitle += "  layer ";
          structtitle += layer;
        }

        TString structandlayer = structure;
        if (layer != 0) {
          if (structure == "TID" || structure == "TEC")
            structandlayer += "_disc";
          else
            structandlayer += "_layer";
          structandlayer += layer;
        }

        for (auto &DMR : DMRs) {
          auto name = get<0>(DMR), ytitle = get<1>(DMR);

          if (name.Contains("plus") || name.Contains("minus") || name.Contains("delta")) {
            if (structname == "TEC" || structname == "TID")
              continue;  //Lorentz drift cannot appear in TEC and TID. These structures are skipped when looking at outward and inward pointing modules.
          }

          cout << bold << name << normal << endl;

          float ymin = get<2>(DMR), ymax = get<3>(DMR);
          Trend trend(Form("%s_%s_%s", Variable.second.get_value<string>().data(), structandlayer.Data(), name.Data()),
                      outputdir.data(),
                      ytitle,
                      ytitle,
                      ymin,
                      ymax,
                      style,
                      GetLumi,
                      lumiAxisType.data());
          trend.lgd.SetHeader(structtitle);

          for (auto const &alignment : alignments) {
            bool fullRange = true;
            if (style.get_child("trends").count("earlyStops")) {
              for (auto const &earlyStop : style.get_child("trends.earlyStops")) {
                if (earlyStop.second.get_value<string>() == alignment.first)
                  fullRange = false;
              }
            }

            TString gtitle = alignment.second.get<string>("title");
            TString gname = Form("%s_%s_%s_%s",
                                 Variable.second.get_value<string>().data(),
                                 gtitle.Data(),
                                 structandlayer.Data(),
                                 name.Data());
            gname.ReplaceAll(" ", "_");
            auto g = Get<TGraphErrors>(gname);
            assert(g != nullptr);
            g->SetTitle(gtitle);  // for the legend
            g->SetMarkerSize(0.6);
            int color = alignment.second.get<int>("color");
            int style = floor(alignment.second.get<double>("style") / 100.);
            g->SetFillColorAlpha(color, 0.2);
            g->SetMarkerColor(color);
            g->SetMarkerStyle(style);
            g->SetLineColor(kWhite);
            trend(g, "P2", "pf", fullRange);
          }
        }
      }
    }
  }

  f->Close();
  cout << bold << "Done" << normal << endl;

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char *argv[]) { return exceptions<trends>(argc, argv); }
#endif
