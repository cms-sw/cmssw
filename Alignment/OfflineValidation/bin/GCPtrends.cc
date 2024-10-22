#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "TFile.h"
#include "TChain.h"
#include "TSelector.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TGraphErrors.h"
#include "TH1D.h"
#include "TObject.h"
#include "TMath.h"
#include "TString.h"

#include "Alignment/OfflineValidation/interface/Trend.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage of the program : GCPtrends [gcp_trend_config.json]" << std::endl;
    std::cout << "gcp_trend_config.json : file with configuration in json format" << std::endl;
    exit(1);
  }

  // parsing input file
  std::string lumi_file(argv[1]);
  boost::property_tree::ptree config_root;
  boost::property_tree::read_json(lumi_file, config_root);

  // configuration
  TString outputdir = config_root.get<std::string>("outputdir", ".");
  outputdir = TString(outputdir) + TString(outputdir.EndsWith("/") ? "" : "/");

  bool usebadmodules = config_root.get<bool>("usebadmodules", true);

  bool splitpixel = config_root.get<bool>("splitpixel", false);

  Int_t trackerphase = config_root.get<int>("trackerphase");

  TString lumitype = config_root.get<std::string>("trends.lumitype");

  Int_t firstrun = config_root.get<int>("trends.firstrun");

  Int_t lastrun = config_root.get<int>("trends.lastrun");

  std::string lumibyrunfile = config_root.get<std::string>("lumibyrunfile", "");
  std::ifstream lumifile(lumibyrunfile);
  assert(lumifile.good());
  const Run2Lumi lumi_per_run(lumibyrunfile, firstrun, lastrun, 1000);

  std::map<TString, Int_t> comparison_map;
  for (boost::property_tree::ptree::value_type &comparison : config_root.get_child("comparisons")) {
    TString path = comparison.first;
    Int_t run_label = comparison.second.get_value<int>();

    comparison_map[path] = run_label;
  }

  // create output file
  TFile *file_out = TFile::Open(outputdir + "GCPTrends.root", "RECREATE");
  file_out->cd();

  gErrorIgnoreLevel = kError;

  // sub-detector mapping
  std::map<Int_t, TString> Sublevel_Subdetector_Map = {
      {1, "PXB"}, {2, "PXF"}, {3, "TIB"}, {4, "TID"}, {5, "TOB"}, {6, "TEC"}};

  // wheels/layers per tracker phase: <pahse,<sublevel,number of layers/wheels>>
  const std::map<Int_t, std::map<Int_t, Int_t>> Phase_Subdetector_Layers_Map = {
      {0, {{1, 3}, {2, 2}}}, {1, {{1, 4}, {2, 3}}}, {2, {{1, 4}, {2, 12}}}};

  // adding layers/wheels in case Pixel is requested further split
  if (splitpixel) {
    assert(trackerphase < 3);

    for (auto &sub_layer : Phase_Subdetector_Layers_Map.at(trackerphase)) {
      for (int iLayer = 1; iLayer <= sub_layer.second; iLayer++) {
        // subid*100+subsubid
        if (sub_layer.first % 2 != 0) {
          Sublevel_Subdetector_Map[100 * sub_layer.first + iLayer] =
              Sublevel_Subdetector_Map[sub_layer.first] + "Layer" + TString(std::to_string(iLayer));

        } else {
          Sublevel_Subdetector_Map[100 * sub_layer.first + (1 - 1) * sub_layer.second + iLayer] =
              Sublevel_Subdetector_Map[sub_layer.first] + "Wheel" + TString(std::to_string(iLayer)) + "Side1";
          Sublevel_Subdetector_Map[100 * sub_layer.first + (2 - 1) * sub_layer.second + iLayer] =
              Sublevel_Subdetector_Map[sub_layer.first] + "Wheel" + TString(std::to_string(iLayer)) + "Side2";
        }
      }
    }
  }

  // variable mapping
  const std::map<Int_t, TString> Index_Variable_Map = {
      {0, "DX"}, {1, "DY"}, {2, "DZ"}, {3, "DAlpha"}, {4, "DBeta"}, {5, "DGamma"}};
  const std::map<TString, TString> Variable_Name_Map = {{"DX", "#Delta x"},
                                                        {"DY", "#Delta y"},
                                                        {"DZ", "#Delta z"},
                                                        {"DAlpha", "#Delta #alpha"},
                                                        {"DBeta", "#Delta #beta"},
                                                        {"DGamma", "#Delta #gamma"}};
  // estimator mapping
  const std::map<Int_t, TString> Index_Estimator_Map = {{0, "Mean"}, {1, "Sigma"}};
  const std::map<TString, TString> Estimator_Name_Map = {{"Mean", "#mu"}, {"Sigma", "#sigma"}};

  // constant unit conversion
  const int convert_cm_to_mum = 10000;
  const int convert_rad_to_mrad = 1000;

  std::cout << std::endl;
  std::cout << "   ==> " << comparison_map.size() << " GCPs to be processed in configuration file ... " << std::endl;
  std::cout << std::endl;

  // booking TGraphs and TH1D
  TH1::StatOverflows(kTRUE);
  std::map<Int_t, std::map<Int_t, std::map<Int_t, TGraphErrors *>>> Graphs;
  std::map<Int_t, std::map<Int_t, TH1D *>> Histos;
  for (auto &level : Sublevel_Subdetector_Map) {
    for (auto &variable : Index_Variable_Map) {
      Histos[level.first][variable.first] = new TH1D(
          "Histo_" + level.second + "_" + variable.second, "Histo_" + level.second + "_" + variable.second, 1, -1, 1);

      for (auto &estimator : Index_Estimator_Map) {
        Graphs[level.first][variable.first][estimator.first] = new TGraphErrors();
      }
    }
  }

  // loop over the comparisons (GCPs)
  for (auto &path_run_map : comparison_map) {
    TChain Events("alignTree", "alignTree");
    Events.Add(path_run_map.first);

    Int_t run_number = path_run_map.second;

    std::cout << "       --> processing GCPtree file: " << path_run_map.first << std::endl;

    TTreeReader Reader(&Events);
    Reader.Restart();

    TTreeReaderValue<Int_t> id = {Reader, "id"};
    TTreeReaderValue<Int_t> badModuleQuality = {Reader, "badModuleQuality"};
    TTreeReaderValue<Int_t> inModuleList = {Reader, "inModuleList"};
    TTreeReaderValue<Int_t> level = {Reader, "level"};
    TTreeReaderValue<Int_t> mid = {Reader, "mid"};
    TTreeReaderValue<Int_t> mlevel = {Reader, "mlevel"};
    TTreeReaderValue<Int_t> sublevel = {Reader, "sublevel"};
    TTreeReaderValue<Float_t> x = {Reader, "x"};
    TTreeReaderValue<Float_t> y = {Reader, "y"};
    TTreeReaderValue<Float_t> z = {Reader, "z"};
    TTreeReaderValue<Float_t> r = {Reader, "r"};
    TTreeReaderValue<Float_t> phi = {Reader, "phi"};
    TTreeReaderValue<Float_t> eta = {Reader, "eta"};
    TTreeReaderValue<Float_t> alpha = {Reader, "alpha"};
    TTreeReaderValue<Float_t> beta = {Reader, "beta"};
    TTreeReaderValue<Float_t> gamma = {Reader, "gamma"};
    TTreeReaderValue<Float_t> dx = {Reader, "dx"};
    TTreeReaderValue<Float_t> dy = {Reader, "dy"};
    TTreeReaderValue<Float_t> dz = {Reader, "dz"};
    TTreeReaderValue<Float_t> dr = {Reader, "dr"};
    TTreeReaderValue<Float_t> dphi = {Reader, "dphi"};
    TTreeReaderValue<Float_t> dalpha = {Reader, "dalpha"};
    TTreeReaderValue<Float_t> dbeta = {Reader, "dbeta"};
    TTreeReaderValue<Float_t> dgamma = {Reader, "dgamma"};
    TTreeReaderValue<Float_t> du = {Reader, "du"};
    TTreeReaderValue<Float_t> dv = {Reader, "dv"};
    TTreeReaderValue<Float_t> dw = {Reader, "dw"};
    TTreeReaderValue<Float_t> da = {Reader, "da"};
    TTreeReaderValue<Float_t> db = {Reader, "db"};
    TTreeReaderValue<Float_t> dg = {Reader, "dg"};
    TTreeReaderValue<Int_t> useDetId = {Reader, "useDetId"};
    TTreeReaderValue<Int_t> detDim = {Reader, "detDim"};
    TTreeReaderArray<Int_t> identifiers = {Reader, "identifiers"};

    // loop over modules
    while (Reader.Next()) {
      if (!usebadmodules)
        if (*badModuleQuality.Get())
          continue;

      Int_t sublevel_idx = *sublevel.Get();

      Histos[sublevel_idx][0]->Fill(*dx.Get() * convert_cm_to_mum);
      Histos[sublevel_idx][1]->Fill(*dy.Get() * convert_cm_to_mum);
      Histos[sublevel_idx][2]->Fill(*dz.Get() * convert_cm_to_mum);
      Histos[sublevel_idx][3]->Fill(*dalpha.Get() * convert_rad_to_mrad);
      Histos[sublevel_idx][4]->Fill(*dbeta.Get() * convert_rad_to_mrad);
      Histos[sublevel_idx][5]->Fill(*dgamma.Get() * convert_rad_to_mrad);

      if (splitpixel && sublevel_idx <= 2) {
        Int_t layer_index;

        if (sublevel_idx % 2 != 0)
          layer_index = 100 * sublevel_idx + identifiers[2];
        else
          layer_index = 100 * sublevel_idx +
                        (identifiers[4] - 1) * Phase_Subdetector_Layers_Map.at(trackerphase).at(sublevel_idx) +
                        identifiers[3];

        Histos[layer_index][0]->Fill(*dx.Get() * convert_cm_to_mum);
        Histos[layer_index][1]->Fill(*dy.Get() * convert_cm_to_mum);
        Histos[layer_index][2]->Fill(*dz.Get() * convert_cm_to_mum);
        Histos[layer_index][3]->Fill(*dalpha.Get() * convert_rad_to_mrad);
        Histos[layer_index][4]->Fill(*dbeta.Get() * convert_rad_to_mrad);
        Histos[layer_index][5]->Fill(*dgamma.Get() * convert_rad_to_mrad);
      }
    }

    for (auto &level : Sublevel_Subdetector_Map) {
      for (auto &variable : Index_Variable_Map) {
        Double_t mean = Histos[level.first][variable.first]->GetMean();
        Double_t sigma = Histos[level.first][variable.first]->GetStdDev();

        Graphs[level.first][variable.first][0]->AddPoint(run_number, mean);
        if (std::fabs(mean) > Graphs[level.first][variable.first][0]->GetMaximum() && std::fabs(mean) > 0.) {
          Graphs[level.first][variable.first][0]->SetMaximum(std::fabs(mean));
          Graphs[level.first][variable.first][0]->SetMinimum(-std::fabs(mean));
        }

        Graphs[level.first][variable.first][1]->AddPoint(run_number, sigma);
        if (sigma > Graphs[level.first][variable.first][1]->GetMaximum() && sigma > 0.) {
          Graphs[level.first][variable.first][1]->SetMaximum(sigma);
          Graphs[level.first][variable.first][1]->SetMinimum(0.);
        }

        Histos[level.first][variable.first]->Reset("ICESM");
      }
    }
  }

  // saving TGraphs
  for (auto &level : Sublevel_Subdetector_Map) {
    for (auto &variable : Index_Variable_Map) {
      for (auto &estimator : Index_Estimator_Map) {
        Graphs[level.first][variable.first][estimator.first]->Write("Graph_" + level.second + "_" + variable.second +
                                                                    "_" + estimator.second);

        TString units = "mrad";
        if (variable.second.Contains("DX") || variable.second.Contains("DY") || variable.second.Contains("DZ"))
          units = "#mum";

        Trend trend("Graph_" + level.second + "_" + variable.second + "_" + estimator.second,
                    "output",
                    "Trend",
                    Estimator_Name_Map.at(estimator.second) + "_{" + Variable_Name_Map.at(variable.second) + "} [" +
                        units + "]",
                    -2 * std::fabs(Graphs[level.first][variable.first][estimator.first]->GetMinimum()),
                    2 * std::fabs(Graphs[level.first][variable.first][estimator.first]->GetMaximum()),
                    config_root,
                    lumi_per_run,
                    lumitype);

        Graphs[level.first][variable.first][estimator.first]->SetFillColor(4);

        trend.lgd.SetHeader(level.second, "center");
        trend.lgd.AddEntry(Graphs[level.first][variable.first][estimator.first], "Average over all modules", "pl");

        trend(Graphs[level.first][variable.first][estimator.first], "p", "pf", false);
      }
    }
  }

  file_out->Close();

  std::cout << std::endl;
  std::cout << "   ==> done." << std::endl;
  std::cout << std::endl;

  return 0;
}
