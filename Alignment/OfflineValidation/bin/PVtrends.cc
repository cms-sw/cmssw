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
#include "fmt/printf.h"

#include "TString.h"
#include "TColor.h"

#include "Alignment/OfflineValidation/interface/PreparePVTrends.h"
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
  bool doRMS = validation.count("doRMS") ? validation.get<bool>("doRMS") : true;
  bool doUnitTest = validation.count("doUnitTest") ? validation.get<bool>("doUnitTest") : false;
  int nWorkers = validation.count("nWorkers") ? validation.get<int>("nWorkers") : 20;
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

  string lumiAxisType = "recorded";
  if (lumiInputFile.Contains("delivered"))
    lumiAxisType = "delivered";

  std::cout << Form("NOTE: using %s luminosity!", lumiAxisType.data()) << std::endl;

  for (auto const &childTreeAlignments : alignments) {
    fs::create_directory(childTreeAlignments.first.c_str());
    for (auto const &childTreeIOV : validation.get_child("IOV")) {
      int iov = childTreeIOV.second.get_value<int>();
      TString file = childTreeAlignments.second.get<string>("file");
      fs::path input_dir = Form("%s/PVValidation_%s_%d.root",
                                file.ReplaceAll("{}", to_string(iov)).Data(),
                                childTreeAlignments.first.c_str(),
                                iov);
      fs::path link_dir = Form(
          "./%s/PVValidation_%s_%d.root", childTreeAlignments.first.c_str(), childTreeAlignments.first.c_str(), iov);
      if (!fs::exists(link_dir) && fs::exists(input_dir))
        fs::create_symlink(input_dir, link_dir);
    }
  }

  string labels_to_add = "";
  if (validation.count("labels")) {
    for (auto const &label : validation.get_child("labels")) {
      labels_to_add += "_";
      labels_to_add += label.second.get_value<string>();
    }
  }

  fs::path pname = Form("%s/PVtrends%s.root", outputdir.data(), labels_to_add.data());

  PreparePVTrends prepareTrends(pname.c_str(), nWorkers, alignments);
  prepareTrends.multiRunPVValidation(doRMS, pathToLumiFile.string().data(), doUnitTest);

  assert(fs::exists(pname));

  float convertUnit = style.get_child("trends").count("convertUnit")
                          ? style.get_child("trends").get<float>("convertUnit")
                          : defaultConvertScale;
  int firstRun = validation.count("firstRun") ? validation.get<int>("firstRun") : startRun2016;
  int lastRun = validation.count("lastRun") ? validation.get<int>("lastRun") : endRun2018;

  const Run2Lumi GetLumi(pathToLumiFile.string().data(), firstRun, lastRun, convertUnit);

  vector<string> variables{"dxy_phi_vs_run", "dxy_eta_vs_run", "dz_phi_vs_run", "dz_eta_vs_run"};

  vector<string> titles{"of impact parameter in transverse plane as a function of azimuth angle",
                        "of impact parameter in transverse plane as a function of pseudorapidity",
                        "of impact parameter along z-axis as a function of azimuth angle",
                        "of impact parameter along z-axis as a function of pseudorapidity"};

  vector<string> ytitles{
      "of d_{xy}(#phi)   [#mum]", "of d_{xy}(#eta)   [#mum]", "of d_{z}(#phi)   [#mum]", "of d_{z}(#eta)   [#mum]"};

  auto f = TFile::Open(pname.c_str());
  for (size_t i = 0; i < variables.size(); i++) {
    Trend mean(Form("mean_%s", variables[i].data()),
               outputdir.data(),
               Form("mean %s", titles[i].data()),
               Form("mean %s", ytitles[i].data()),
               -7.,
               10.,
               style,
               GetLumi,
               lumiAxisType.data());
    Trend RMS(Form("RMS_%s", variables[i].data()),
              outputdir.data(),
              Form("RMS %s", titles[i].data()),
              Form("RMS %s", ytitles[i].data()),
              0.,
              35.,
              style,
              GetLumi,
              lumiAxisType.data());
    mean.lgd.SetHeader("p_{T} (track) > 3 GeV");
    RMS.lgd.SetHeader("p_{T} (track) > 3 GeV");

    for (auto const &alignment : alignments) {
      bool fullRange = true;
      if (style.get_child("trends").count("earlyStops")) {
        for (auto const &earlyStop : style.get_child("trends.earlyStops")) {
          if (earlyStop.second.get_value<string>() == alignment.first)
            fullRange = false;
        }
      }

      TString gname = alignment.second.get<string>("title");
      gname.ReplaceAll(" ", "_");

      auto gMean = Get<TGraph>(fmt::sprintf("mean_%s_%s", gname.Data(), variables[i].data()).c_str());
      auto hRMS = Get<TH1>(fmt::sprintf("RMS_%s_%s", gname.Data(), variables[i].data()).c_str());
      assert(gMean != nullptr);
      assert(hRMS != nullptr);

      TString gtitle = alignment.second.get<string>("title");
      gMean->SetTitle(gtitle);  // for the legend
      //gMean->SetTitle(""); // for the legend
      gMean->SetMarkerSize(0.6);
      int color = alignment.second.get<int>("color");
      int style = floor(alignment.second.get<double>("style") / 100.);
      gMean->SetMarkerColor(color);
      gMean->SetMarkerStyle(style);

      hRMS->SetTitle(gtitle);  // for the legend
      //hRMS ->SetTitle(""); // for the legend
      hRMS->SetMarkerSize(0.6);
      hRMS->SetMarkerColor(color);
      hRMS->SetMarkerStyle(style);

      mean(gMean, "PZ", "p", fullRange);
      RMS(hRMS, "", "p", fullRange);
    }
  }

  f->Close();
  cout << bold << "Done" << normal << endl;

  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char *argv[]) { return exceptions<trends>(argc, argv); }
#endif
