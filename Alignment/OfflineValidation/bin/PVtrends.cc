#include <cstdlib>
#include <string>
#include <tuple>
#include <iostream>
#include <numeric>
#include <functional>
#include <unistd.h>

#include <TFile.h>
#include <TGraph.h>
#include <TH1.h>

#include "exceptions.h"
#include "toolbox.h"
#include "Options.h"

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>

#include "TString.h"
#include "TColor.h"

#include "Alignment/OfflineValidation/interface/PreparePVTrends.h"
#include "Alignment/OfflineValidation/interface/Trend.h"

using namespace std;
using namespace AllInOneConfig;
namespace fs = boost::filesystem;

static const char * bold = "\e[1m", * normal = "\e[0m";

namespace pt = boost::property_tree;

int trends(int argc, char* argv[]) {

  // parse the command line

  Options options;
  options.helper(argc, argv);
  options.parser(argc, argv);
  
  //Read in AllInOne json config
  pt::ptree main_tree;
  pt::read_json(options.config, main_tree);
  
  pt::ptree alignments = main_tree.get_child("alignments");
  pt::ptree validation = main_tree.get_child("validation");
  pt::ptree lines = main_tree.get_child("lines");
  
  //Read all configure variables and set default for missing keys
  string outputdir = main_tree.get<string>("output");
  bool doRMS = validation.get_child_optional("doRMS") ? validation.get<bool>("doRMS") : true;
  bool fullRange = validation.get_child_optional("fullRange") ? validation.get<bool>("fullRange") : false;
  bool doUnitTest = validation.get_child_optional("doUnitTest") ? validation.get<bool>("doUnitTest") : false;
  TString lumiInputFile = validation.get_child_optional("lumiInputFile") ? validation.get<string>("lumiInputFile") : "lumiperFullRun2_delivered.txt";

  vector<string> labels{};
  if (validation.get_child_optional("labels")) {
    labels.clear();
    for (const pair<string, pt::ptree>& childTree : validation.get_child("labels")) {
      labels.push_back(childTree.second.get_value<string>());
    }
  }

  TString LumiFile = getenv("CMSSW_BASE");
  if (lumiInputFile.BeginsWith("/"))
    LumiFile = lumiInputFile;
  else {
    LumiFile += "/src/Alignment/OfflineValidation/data/";
    LumiFile += lumiInputFile;
  }
  fs::path pathToLumiFile = LumiFile.Data();
  if (!fs::exists(pathToLumiFile)) {
    cout << "ERROR: lumi-per-run file (" << LumiFile.Data() << ") not found!" << endl << "Please check!" << endl;
    exit(EXIT_FAILURE);
  }

  for (const pair<string, pt::ptree>& childTreeAlignments : alignments) {
    fs::create_directory(childTreeAlignments.first.c_str());
    for (const pair<string, pt::ptree>& childTreeIOV : validation.get_child("IOV")) {
      int iov = childTreeIOV.second.get_value<int>();
      TString file = childTreeAlignments.second.get<string>("file");
      fs::path input_dir = Form("%s/PVValidation_%s_%d.root", file.ReplaceAll("{}", to_string(iov)).Data(), childTreeAlignments.first.c_str(), iov);
      fs::path link_dir = Form("./%s/PVValidation_%s_%d.root", childTreeAlignments.first.c_str(), childTreeAlignments.first.c_str(), iov);
      if(!fs::exists(link_dir) && fs::exists(input_dir))
	fs::create_symlink(input_dir, link_dir);
    }
  }

  PreparePVTrends prepareTrends(outputdir, alignments);
  prepareTrends.MultiRunPVValidation(labels, doRMS, LumiFile, doUnitTest);

  Trend::CMS = "#scale[1.1]{#bf{CMS}}";

  int firstRun = validation.get_child_optional("firstRun") ? validation.get<int>("firstRun") : 272930;
  int lastRun = validation.get_child_optional("lastRun") ? validation.get<int>("lastRun") : 325175;
  
  const Run2Lumi GetLumi(LumiFile.Data(), firstRun, lastRun);

  string labels_to_add = "";
  if (labels.size() != 0 ) {
    for (const auto &label : labels) {
      labels_to_add += "_";
      labels_to_add += label;
    }
  }
  fs::path pname = Form("%s/PVtrends%s.root", outputdir.data(), labels_to_add.data());
  assert(fs::exists(pname));

  vector<string> variables{"dxy_phi_vs_run",
                            "dxy_eta_vs_run",
                            "dz_phi_vs_run",
                            "dz_eta_vs_run"};
  
  
  vector<string> titles{"of impact parameter in transverse plane as a function of azimuth angle",
  		       "of impact parameter in transverse plane as a function of pseudorapidity",
                       "of impact parameter along z-axis as a function of azimuth angle",
                       "of impact parameter along z-axis as a function of pseudorapidity"};
  
  vector<string> ytitles{"of d_{xy}(#phi)   [#mum]",
                        "of d_{xy}(#eta)   [#mum]",
                        "of d_{z}(#phi)   [#mum]",
                        "of d_{z}(#eta)   [#mum]"};
  
  auto f = TFile::Open(pname.c_str());
  for(size_t i=0; i<variables.size(); i++) {
  
    Trend mean(Form("mean_%s", variables[i].data()), outputdir.data(), Form("mean %s", titles[i].data()),
  	       Form("mean %s", ytitles[i].data()), -7., 10., lines, GetLumi), 
          RMS (Form("RMS_%s", variables[i].data()), outputdir.data(), Form("RMS %s", titles[i].data()), 
  	       Form("RMS %s", ytitles[i].data()), 0., 35., lines, GetLumi);
    mean.lgd.SetHeader("p_{T} (track) > 3 GeV");
    RMS.lgd.SetHeader("p_{T} (track) > 3 GeV");
  
    for (const pair<string, pt::ptree>& childTree : alignments) {
      TString gname = childTree.second.get<string>("title");
      gname.ReplaceAll(" ", "_");
    
      auto gMean = Get<TGraph>("mean_%s_%s", gname.Data(), variables[i].data());
      auto hRMS  = Get<TH1   >( "RMS_%s_%s", gname.Data(), variables[i].data());
      assert(gMean != nullptr);
      assert(hRMS  != nullptr);
      
      TString gtitle = childTree.second.get<string>("title");
      //gMean->SetTitle(gtitle); // for the legend
      gMean->SetTitle(""); // for the legend
      gMean->SetMarkerSize(0.6);
      int color = childTree.second.get<int>("color");
      int style = childTree.second.get<int>("style");
      gMean->SetMarkerColor(color);
      gMean->SetMarkerStyle(style);
  
      //hRMS ->SetTitle(gtitle); // for the legend
      hRMS ->SetTitle(""); // for the legend
      hRMS ->SetMarkerSize(0.6);
      hRMS ->SetMarkerColor(color);
      hRMS ->SetMarkerStyle(style);
    
      mean(gMean, "PZ", "p", fullRange);
      RMS (hRMS , ""  , "p", fullRange);

      // dirty trick to get larger marker in the legend TODO??
      double x[] = {-99};
      auto g2 = new TGraph(1,x,x);
      g2->SetTitle(gtitle); // for the legend
      g2->SetMarkerColor(color);
      g2->SetMarkerStyle(style);
      mean(g2, "PZ", "p", false);
      RMS (g2, "PZ", "p", false);
    }
  }
  
  f->Close();
  cout << bold << "Done" << normal << endl;
  
  return EXIT_SUCCESS;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
int main(int argc, char* argv[]) { return exceptions<trends>(argc, argv); }
#endif
