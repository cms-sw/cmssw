#include "Alignment/OfflineValidation/interface/PrepareDMRTrends.h"

using namespace std;
namespace fs = std::experimental::filesystem;
namespace pt = boost::property_tree;

PrepareDMRTrends::PrepareDMRTrends(const char *outputFileName, pt::ptree &json) : outputFileName_(outputFileName) {
  geometries.clear();
  for (const auto &childTree : json) {
    geometries.push_back(childTree.second.get<std::string>("title"));
  }
}

/*! \fn getName
 *  \brief Function used to get a string containing information on the high level structure, the layer/disc and the geometry.
 */

TString PrepareDMRTrends::getName(TString structure, int layer, TString geometry) {
  geometry.ReplaceAll(" ", "_");
  TString name = geometry + "_" + structure;
  if (layer != 0) {
    if (structure == "TID" || structure == "TEC")
      name += "_disc";
    else
      name += "_layer";
    name += layer;
  }

  return name;
};

/*! \fn compileDMRTrends
 *  \brief  Create a file where the DMR trends are stored in the form of TGraph.
 */

void PrepareDMRTrends::compileDMRTrends(vector<int> IOVlist,
                                        TString Variable,
                                        std::vector<std::string> inputFiles,
                                        vector<TString> structures,
                                        const map<TString, int> nlayers,
                                        bool FORCE) {
  gROOT->SetBatch();

  float ScaleFactor = DMRFactor;
  if (Variable == "DrmsNR")
    ScaleFactor = 1;

  map<pair<pair<TString, int>, TString>, Geometry> mappoints;  // pair = (structure, layer), geometry
  Point *point = nullptr;
  TFile *f = nullptr;

  for (unsigned int i = 0; i < inputFiles.size(); ++i) {
    if (fs::is_empty(inputFiles.at(i).c_str())) {
      cout << "ERROR: Empty file " << inputFiles.at(i).c_str() << endl;
      continue;
    }

    int runN = IOVlist.at(i);

    f = new TFile(inputFiles.at(i).c_str(), "READ");
    std::cout << inputFiles.at(i) << std::endl;

    for (TString &structure : structures) {
      TString structname = structure;
      structname.ReplaceAll("_y", "");
      size_t layersnumber = nlayers.at(structname);
      for (size_t layer = 0; layer <= layersnumber; layer++) {
        for (const string &geometry : geometries) {
          TString name = Variable + "_" + getName(structure, layer, geometry);
          TH1F *histo = dynamic_cast<TH1F *>(f->Get(name));

          // Three possibilities:
          //  - All histograms are produced correctly
          //  - Only the non-split histograms are produced
          //  - No histogram is produced correctly
          //  FORCE means that the Point is not added to the points collection in the chosen geometry for that structure
          //  If FORCE is not enabled a default value for the Point is used (-9999) which will appear in the plots
          if (!histo) {
            //cout << "Run" << runN << " Histogram: " << name << " not found" << endl;
            if (FORCE)
              continue;
            point = new Point(runN, ScaleFactor);
          } else if (structure != "TID" && structure != "TEC") {
            TH1F *histoplus = dynamic_cast<TH1F *>(f->Get((name + "_plus")));
            TH1F *histominus = dynamic_cast<TH1F *>(f->Get((name + "_minus")));
            if (!histoplus || !histominus) {
              //cout << "Run" << runN << " Histogram: " << name << " plus or minus not found" << endl;
              if (FORCE)
                continue;
              point = new Point(runN, ScaleFactor, histo);
            } else
              point = new Point(runN, ScaleFactor, histo, histoplus, histominus);

          } else
            point = new Point(runN, ScaleFactor, histo);
          mappoints[make_pair(make_pair(structure, layer), geometry)].points.push_back(*point);
        }
      }
    }
    f->Close();
  }

  TFile *fout = TFile::Open(outputFileName_, "RECREATE");
  TGraphErrors *g = nullptr;
  for (TString &structure : structures) {
    TString structname = structure;
    structname.ReplaceAll("_y", "");
    size_t layersnumber = nlayers.at(structname);
    for (size_t layer = 0; layer <= layersnumber; layer++) {
      for (const string &geometry : geometries) {
        TString name = Variable + "_" + getName(structure, layer, geometry);
        Geometry geom = mappoints[make_pair(make_pair(structure, layer), geometry)];
        using Trend = vector<float> (Geometry::*)() const;
        vector<Trend> trends{&Geometry::Mu,
                             &Geometry::Sigma,
                             &Geometry::MuPlus,
                             &Geometry::SigmaPlus,
                             &Geometry::MuMinus,
                             &Geometry::SigmaMinus,
                             &Geometry::DeltaMu,
                             &Geometry::SigmaDeltaMu};
        vector<TString> variables{
            "mu", "sigma", "muplus", "sigmaplus", "muminus", "sigmaminus", "deltamu", "sigmadeltamu"};
        vector<float> runs = geom.Run();
        size_t n = runs.size();
        vector<float> emptyvec;
        for (size_t i = 0; i < runs.size(); i++)
          emptyvec.push_back(0.);
        for (size_t iVar = 0; iVar < variables.size(); iVar++) {
          Trend trend = trends.at(iVar);
          g = new TGraphErrors(n, runs.data(), (geom.*trend)().data(), emptyvec.data(), emptyvec.data());
          g->SetTitle(geometry.c_str());
          g->Write(name + "_" + variables.at(iVar));
        }
        vector<pair<Trend, Trend>> trendspair{make_pair(&Geometry::Mu, &Geometry::Sigma),
                                              make_pair(&Geometry::MuPlus, &Geometry::SigmaPlus),
                                              make_pair(&Geometry::MuMinus, &Geometry::SigmaMinus),
                                              make_pair(&Geometry::DeltaMu, &Geometry::SigmaDeltaMu)};
        vector<pair<TString, TString>> variablepairs{make_pair("mu", "sigma"),
                                                     make_pair("muplus", "sigmaplus"),
                                                     make_pair("muminus", "sigmaminus"),
                                                     make_pair("deltamu", "sigmadeltamu")};
        for (size_t iVar = 0; iVar < variablepairs.size(); iVar++) {
          Trend meantrend = trendspair.at(iVar).first;
          Trend sigmatrend = trendspair.at(iVar).second;
          g = new TGraphErrors(
              n, runs.data(), (geom.*meantrend)().data(), emptyvec.data(), (geom.*sigmatrend)().data());
          g->SetTitle(geometry.c_str());
          TString graphname = name + "_" + variablepairs.at(iVar).first;
          graphname += variablepairs.at(iVar).second;
          g->Write(graphname);
        }
      }
    }
  }
  fout->Close();

  delete point;
  delete f;
  delete g;
}
