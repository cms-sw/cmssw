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

#include "Alignment/OfflineValidation/interface/PrepareDMRTrends.h"
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
  bool fullRange = validation.get_child_optional("fullRange") ? validation.get<bool>("fullRange") : false;
  bool FORCE = validation.get_child_optional("FORCE") ? validation.get<bool>("FORCE") : false;
  string Year = validation.get_child_optional("Year") ? validation.get<string>("Year") : "Run2";
  TString lumiInputFile = validation.get_child_optional("lumiInputFile") ? validation.get<string>("lumiInputFile") : "lumiperFullRun2_delivered.txt";

  vector<string> labels{};
  if (validation.get_child_optional("labels")) {
    labels.clear();
    for (const pair<string, pt::ptree>& childTree : validation.get_child("labels")) {
      labels.push_back(childTree.second.get_value<string>());
    }
  }
  
  vector<string> Variables;
  if (validation.get_child_optional("Variables")) {
    for (const pair<string, pt::ptree>& childTree : validation.get_child("Variables")) {
      Variables.push_back(childTree.second.get_value<string>());
    }
  }
  else
    Variables.push_back("median");

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

  vector<int> IOVlist;
  vector<string> inputFiles;
  for (const pair<string, pt::ptree>& childTree : validation.get_child("IOV")) {
    int iov = childTree.second.get_value<int>();
    IOVlist.push_back(iov);
    TString mergeFile =  validation.get<string>("mergeFile");
    string input = Form("%s/OfflineValidationSummary.root", mergeFile.ReplaceAll("{}", to_string(iov)).Data());
    inputFiles.push_back(input);
  }

  PrepareDMRTrends prepareTrends(outputdir, alignments);
  for (const auto &Variable : Variables) {
    prepareTrends.compileDMRTrends(IOVlist, Variable, labels, Year, inputFiles, FORCE);
  }

  Trend::CMS = "#scale[1.1]{#bf{CMS}}";

  vector<TString> structures{"BPIX", "BPIX_y", "FPIX", "FPIX_y", "TIB", "TID", "TOB", "TEC"};

  map<TString, int> nlayers{{"BPIX", 4}, {"FPIX", 3}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};
  if (Year == "2016")
    nlayers = {{"BPIX", 3}, {"FPIX", 2}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};
  
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
  fs::path pname = Form("%s/DMRtrends%s.root", outputdir.data(), labels_to_add.data());
  assert(fs::exists(pname));
  
  auto f = TFile::Open(pname.c_str());

  for (auto Variable : Variables) {

    vector<tuple<TString, TString, float, float>> DMRs {
      {"mu", "#mu [#mum]", -6, 6},
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
      {"deltamusigmadeltamu", "#Delta#mu [#mum]", -15, 15}
    };

    if (Variable == "DrmsNR") {
      DMRs = {
	{"mu", "RMS(x'_{pred}-x'_{hit} /#sigma)", -1.2, 1.2},
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
	{"deltamusigmadeltamu", "#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)", -0.15, 0.15}
      };
    }

    for (TString &structure : structures) {
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
	
        for (auto DMR: DMRs) {
	  
          auto name  = get<0>(DMR),
            ytitle = get<1>(DMR);
      	  
          if (name.Contains("plus") || name.Contains("minus") || name.Contains("delta")) {
	    if (structname == "TEC" || structname == "TID")
	      continue;  //Lorentz drift cannot appear in TEC and TID. These structures are skipped when looking at outward and inward pointing modules.
          }
          
          cout << bold << name << normal << endl;
          
          float ymin = get<2>(DMR), ymax = get<3>(DMR);
          Trend trend(Form("%s_%s_%s", Variable.data(), structandlayer.Data(), name.Data()), outputdir.data(), ytitle, ytitle, ymin, ymax, lines, GetLumi);
          trend.lgd.SetHeader(structtitle);
          
          for (const pair<string, pt::ptree>& childTree : alignments) {
            TString alignment = childTree.second.get<string>("title");
            TString gname = Form("%s_%s_%s_%s", Variable.data(), alignment.Data(), structandlayer.Data(), name.Data());
            gname.ReplaceAll(" ", "_");
            auto g = Get<TGraphErrors>(gname);
            assert(g != nullptr);
            g->SetTitle(""); // for the legend
            g->SetMarkerSize(0.6);
            int color = childTree.second.get<int>("color");
            int style = childTree.second.get<int>("style");
            g->SetFillColorAlpha(color, 0.2);
            g->SetMarkerColor(color);
            g->SetMarkerStyle(style);
            g->SetLineColor(kWhite);
            trend(g, "P2", "pf", fullRange);
            
            // dirty trick to get bigger marker in the legend 
            double x[] = {-99};
            auto g2 = new TGraph(1,x,x);
            g2->SetTitle(alignment);
            g2->SetMarkerColor(color);
            g2->SetFillColorAlpha(color, 0.2);
            g2->SetLineColor(kWhite);
            g2->SetMarkerStyle(style);
            trend(g2, "P2", "pf", false);
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
int main(int argc, char* argv[]) { return exceptions<trends>(argc, argv); }
#endif
