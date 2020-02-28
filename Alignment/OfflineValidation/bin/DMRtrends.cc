#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iomanip>
#include <experimental/filesystem>
#include "TPad.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TMultiGraph.h"
#include "TH1.h"
#include "THStack.h"
#include "TROOT.h"
#include "TFile.h"
#include "TColor.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TMath.h"
#include "TRegexp.h"
#include "TPaveLabel.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TLine.h"
#include "Alignment/OfflineValidation/plugins/ColorParser.C"

using namespace std;
namespace fs = std::experimental::filesystem;

/*!
 * \def Dummy value in case a DMR would fail for instance
 */
#define DUMMY -999.
/*!
 * \def Scale factor value to have luminosity expressed in fb^-1
 */
#define lumiFactor 1000.
/*!
 * \def Scale factor value to have mean and sigmas expressed in micrometers.
 */
#define DMRFactor 10000.

/*! \struct Point
 *  \brief Structure Point
 *         Contains parameters of Gaussian fits to DMRs
 *  
 * @param run:             run number (IOV boundary)
 * @param scale:           scale for the measured quantity: cm->μm for DMRs, 1 for normalized residuals
 * @param mu:              mu/mean from Gaussian fit to DMR/DrmsNR
 * @param sigma:           sigma/standard deviation from Gaussian fit to DMR/DrmsNR
 * @param muplus:          mu/mean for the inward pointing modules
 * @param muminus:         mu/mean for outward pointing modules
 * @param sigmaplus:       sigma/standard for inward pointing modules 
 * @param sigmaminus: //!< sigma/standard for outward pointing modules
 */
struct Point {
  float run, scale, mu, sigma, muplus, muminus, sigmaplus, sigmaminus;

  /*! \fn Point
     *  \brief Constructor of structure Point, initialising all members one by one
     */
  Point(float Run = DUMMY,
        float ScaleFactor = DMRFactor,
        float y1 = DUMMY,
        float y2 = DUMMY,
        float y3 = DUMMY,
        float y4 = DUMMY,
        float y5 = DUMMY,
        float y6 = DUMMY)
      : run(Run), scale(ScaleFactor), mu(y1), sigma(y2), muplus(y3), muminus(y5), sigmaplus(y4), sigmaminus(y6) {}

  /*! \fn Point
     *  \brief Constructor of structure Point, initialising all members from DMRs directly (with split)
     */
  Point(float Run, float ScaleFactor, TH1 *histo, TH1 *histoplus, TH1 *histominus)
      : Point(Run,
              ScaleFactor,
              histo->GetMean(),
              histo->GetMeanError(),
              histoplus->GetMean(),
              histoplus->GetMeanError(),
              histominus->GetMean(),
              histominus->GetMeanError()) {}

  /*! \fn Point
     *  \brief Constructor of structure Point, initialising all members from DMRs directly (without split)
     */
  Point(float Run, float ScaleFactor, TH1 *histo) : Point(Run, ScaleFactor, histo->GetMean(), histo->GetMeanError()) {}

  Point &operator=(const Point &p) {
    run = p.run;
    mu = p.mu;
    muplus = p.muplus;
    muminus = p.muminus;
    sigma = p.sigma;
    sigmaplus = p.sigmaplus;
    sigmaminus = p.sigmaminus;
    return *this;
  }

  float GetRun() const { return run; }
  float GetMu() const { return scale * mu; }
  float GetMuPlus() const { return scale * muplus; }
  float GetMuMinus() const { return scale * muminus; }
  float GetSigma() const { return scale * sigma; }
  float GetSigmaPlus() const { return scale * sigmaplus; }
  float GetSigmaMinus() const { return scale * sigmaminus; }
  float GetDeltaMu() const {
    if (muplus == DUMMY && muminus == DUMMY)
      return DUMMY;
    else
      return scale * (muplus - muminus);
  }
  float GetSigmaDeltaMu() const {
    if (sigmaplus == DUMMY && sigmaminus == DUMMY)
      return DUMMY;
    else
      return scale * hypot(sigmaplus, sigmaminus);
  };
};

///**************************
///*  Function declaration  *
///**************************

TString getName(TString structure, int layer, TString geometry);
TH1F *ConvertToHist(TGraphErrors *g);
const map<TString, int> numberOfLayers(TString Year = "2018");
vector<int> runlistfromlumifile(TString Year = "2018");
bool checkrunlist(vector<int> runs, vector<int> IOVlist = {}, TString Year = "2018");
TString lumifileperyear(TString Year = "2018", string RunOrIOV = "IOV");
void scalebylumi(TGraphErrors *g, vector<pair<int, double>> lumiIOVpairs);
vector<pair<int, double>> lumiperIOV(vector<int> IOVlist, TString Year = "2018");
double getintegratedlumiuptorun(int run, TString Year = "2018", double min = 0.);
void PixelUpdateLines(TCanvas *c,
                      TString Year = "2018",
                      bool showlumi = false,
                      vector<int> pixelupdateruns = {314881, 316758, 317527, 318228, 320377});
void PlotDMRTrends(
    vector<int> IOVlist,
    TString Variable = "median",
    vector<string> labels = {"MB"},
    TString Year = "2018",
    string myValidation = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/results/acardini/DMRs/",
    vector<string> geometries = {"GT", "SG", "MP pix LBL", "PIX HLS+ML STR fix"},
    vector<Color_t> colours = {kBlue, kRed, kGreen, kCyan},
    TString outputdir =
        "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/alignmentObjects/acardini/DMRsTrends/",
    bool pixelupdate = false,
    vector<int> pixelupdateruns = {314881, 316758, 317527, 318228, 320377},
    bool showlumi = false);
void compileDMRTrends(
    vector<int> IOVlist,
    TString Variable = "median",
    vector<string> labels = {"MB"},
    TString Year = "2018",
    string myValidation = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/results/acardini/DMRs/",
    vector<string> geometries = {"GT", "SG", "MP pix LBL", "PIX HLS+ML STR fix"},
    bool showlumi = false,
    bool FORCE = false);
void DMRtrends(
    vector<int> IOVlist,
    vector<string> Variables = {"median", "DrmsNR"},
    vector<string> labels = {"MB"},
    TString Year = "2018",
    string myValidation = "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/results/acardini/DMRs/",
    vector<string> geometries = {"GT", "SG", "MP pix LBL", "PIX HLS+ML STR fix"},
    vector<Color_t> colours = {kBlue, kRed, kGreen, kCyan},
    TString outputdir =
        "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/alignmentObjects/acardini/DMRsTrends/",
    bool pixelupdate = false,
    vector<int> pixelupdateruns = {314881, 316758, 317527, 318228, 320377},
    bool showlumi = false,
    bool FORCE = false);

/*! \class Geometry
 *  \brief Class Geometry
 *         Contains vector for fit parameters (mean, sigma, etc.) obtained from multiple IOVs
 *         See Structure Point for description of the parameters.
 */

class Geometry {
public:
  vector<Point> points;

private:
  //template<typename T> vector<T> GetQuantity (T (Point::*getter)() const) const {
  vector<float> GetQuantity(float (Point::*getter)() const) const {
    vector<float> v;
    for (Point point : points) {
      float value = (point.*getter)();
      v.push_back(value);
    }
    return v;
  }

public:
  TString title;
  Geometry() : title("") {}
  Geometry(TString Title) : title(Title) {}
  Geometry &operator=(const Geometry &geom) {
    title = geom.title;
    points = geom.points;
    return *this;
  }
  void SetTitle(TString Title) { title = Title; }
  TString GetTitle() { return title; }
  vector<float> Run() const { return GetQuantity(&Point::GetRun); }
  vector<float> Mu() const { return GetQuantity(&Point::GetMu); }
  vector<float> MuPlus() const { return GetQuantity(&Point::GetMuPlus); }
  vector<float> MuMinus() const { return GetQuantity(&Point::GetMuMinus); }
  vector<float> Sigma() const { return GetQuantity(&Point::GetSigma); }
  vector<float> SigmaPlus() const { return GetQuantity(&Point::GetSigmaPlus); }
  vector<float> SigmaMinus() const { return GetQuantity(&Point::GetSigmaMinus); }
  vector<float> DeltaMu() const { return GetQuantity(&Point::GetDeltaMu); }
  vector<float> SigmaDeltaMu() const { return GetQuantity(&Point::GetSigmaDeltaMu); }
  //vector<float> Graph (string variable) const {
  // };
};

/// DEPRECATED
//struct Layer {
//    map<string,Geometry> geometries;
//};
//
//struct HLS {
//    vector<Layer> layers;
//    map<string,Geometry> geometries;
//};

/*! \fn getName
 *  \brief Function used to get a string containing information on the high level structure, the layer/disc and the geometry.
 */

TString getName(TString structure, int layer, TString geometry) {
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

/*! \fn numberOfLayers
 *  \brief Function used to retrieve a map containing the number of layers per subdetector
 */

const map<TString, int> numberOfLayers(TString Year) {
  if (Year == "2016")
    return {{"BPIX", 3}, {"FPIX", 2}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};
  else
    return {{"BPIX", 4}, {"FPIX", 3}, {"TIB", 4}, {"TID", 3}, {"TOB", 6}, {"TEC", 9}};
}

/// TO DO: once the information on the luminosity is passed through the root files this method needs to be changed
/*! \fn lumifileperyear
 *  \brief Function to retrieve the file with luminosity per run/IOV
 *         The use of a lumi-per-IOV file is deprecated, but can still be useful for debugging
 */

TString lumifileperyear(TString Year, string RunOrIOV) {
  TString LumiFile = std::getenv("CMSSW_BASE");
  LumiFile += "/src/Alignment/OfflineValidation/data/lumiper";
  if (RunOrIOV != "run" && RunOrIOV != "IOV") {
    cout << "ERROR: Please specify \"run\" or \"IOV\" to retrieve the luminosity run by run or for each IOV" << endl;
    exit(EXIT_FAILURE);
  }
  LumiFile += RunOrIOV;
  if (Year != "2016" && Year != "2017" && Year != "2018") {
    cout << "ERROR: Only 2016, 2017 and 2018 lumi-per-run files are available, please check!" << endl;
    exit(EXIT_FAILURE);
  }
  LumiFile += Year;
  LumiFile += ".txt";
  return LumiFile;
};

/*! \fn runlistfromlumifile
 *  \brief Get a vector containing the list of runs for which the luminosity is known.
 */

vector<int> runlistfromlumifile(TString Year) {
  TString filename = lumifileperyear(Year, "run");
  fs::path path(filename.Data());
  if (fs::is_empty(path)) {
    cout << "ERROR: Empty file " << path.c_str() << endl;
    exit(EXIT_FAILURE);
  }
  TGraph *scale = new TGraph(filename.Data());
  double *xscale = scale->GetX();
  size_t N = scale->GetN();
  vector<int> runs;
  for (size_t i = 0; i < N; i++)
    runs.push_back(xscale[i]);
  return runs;
}

/*! \fn checkrunlist
 *  \brief Check whether all runs of interest are present in the luminosity per run txt file and whether all IOVs analized have been correctly processed
 */

bool checkrunlist(vector<int> runs, vector<int> IOVlist, TString Year) {
  vector<int> runlist = runlistfromlumifile(Year);
  vector<int> missingruns;  //runs for which the luminosity is not found
  vector<int> lostruns;     //IOVs for which the DMR were not found
  bool problemfound = false;
  for (int run : IOVlist) {
    if (find(runlist.begin(), runlist.end(), run) == runlist.end()) {
      problemfound = true;
      missingruns.push_back(run);
    }
  }
  if (!IOVlist.empty())
    for (int IOV : IOVlist) {
      if (find(runs.begin(), runs.end(), IOV) == runs.end()) {
        problemfound = true;
        lostruns.push_back(IOV);
      }
    }
  std::sort(missingruns.begin(), missingruns.end());
  if (problemfound) {
    if (!lostruns.empty()) {
      cout << "WARNING: some IOVs where not found among the list of available DMRs" << endl
           << "List of missing IOVs:" << endl;
      for (int lostrun : lostruns)
        cout << to_string(lostrun) << " ";
      cout << endl;
    }
    if (!missingruns.empty()) {
      cout << "WARNING: some IOVs are missing in the run/luminosity txt file" << endl
           << "List of missing runs:" << endl;
      for (int missingrun : missingruns)
        cout << to_string(missingrun) << " ";
      cout << endl;
    }
  }
  return problemfound;
}

/*! \fn DMRtrends
 *  \brief Create and plot the DMR trends.
 */

void DMRtrends(vector<int> IOVlist,
               vector<string> Variables,
               vector<string> labels,
               TString Year,
               string myValidation,
               vector<string> geometries,
               vector<Color_t> colours,
               TString outputdir,
               bool pixelupdate,
               vector<int> pixelupdateruns,
               bool showlumi,
               bool FORCE) {
  fs::path path(outputdir.Data());
  if (!(fs::exists(path))) {
    cout << "WARNING: Output directory (" << outputdir.Data() << ") not found, it will be created automatically!"
         << endl;
    //	exit(EXIT_FAILURE);
    fs::create_directory(path);
    if (!(fs::exists(path))) {
      cout << "ERROR: Output directory (" << outputdir.Data() << ") has not been created!" << endl
           << "At least the parent directory needs to exist, please check!" << endl;
      exit(EXIT_FAILURE);
    }
  }
  for (TString Variable : Variables) {
    compileDMRTrends(IOVlist, Variable, labels, Year, myValidation, geometries, showlumi, FORCE);
    cout << "Begin plotting" << endl;
    PlotDMRTrends(IOVlist,
                  Variable,
                  labels,
                  Year,
                  myValidation,
                  geometries,
                  colours,
                  outputdir,
                  pixelupdate,
                  pixelupdateruns,
                  showlumi);
  }
};

/*! \fn compileDMRTrends
 *  \brief  Create a file where the DMR trends are stored in the form of TGraph.
 */

void compileDMRTrends(vector<int> IOVlist,
                      TString Variable,
                      vector<string> labels,
                      TString Year,
                      string myValidation,
                      vector<string> geometries,
                      bool showlumi,
                      bool FORCE) {
  gROOT->SetBatch();
  vector<int> RunNumbers;
  vector<TString> filenames;
  TRegexp regexp("[0-9][0-9][0-9][0-9][0-9][0-9]");
  for (const auto &entry : fs::recursive_directory_iterator(myValidation)) {
    bool found_all_labels = true;
    for (string label : labels) {
      if (entry.path().string().find(label) == std::string::npos)
        found_all_labels = false;
    }
    if ((entry.path().string().find("ExtendedOfflineValidation_Images/OfflineValidationSummary.root") !=
         std::string::npos) &&
        found_all_labels) {
      if (fs::is_empty(entry.path()))
        cout << "ERROR: Empty file " << entry.path() << endl;
      else {
        TString filename(entry.path().string());
        filenames.push_back(filename);
        TString runstring(filename(regexp));
        if (runstring.IsFloat()) {
          int runN = runstring.Atoi();
          RunNumbers.push_back(runN);
        }
      }
    }
  }
  if (RunNumbers.empty()) {
    cout << "ERROR: No available DMRs found!" << endl
         << "Please check that the OfflineValidationSummary.root file is in any directory where the DMR per IOV have "
            "been stored!"
         << endl;
    exit(EXIT_FAILURE);
  } else if (checkrunlist(RunNumbers, IOVlist, Year)) {
    cout << "Please check the DMRs that have been produced!" << endl;
    if (!FORCE)
      exit(EXIT_FAILURE);
  }

  vector<TString> structures{"BPIX", "BPIX_y", "FPIX", "FPIX_y", "TIB", "TID", "TOB", "TEC"};

  const map<TString, int> nlayers = numberOfLayers(Year);

  float ScaleFactor = DMRFactor;
  if (Variable == "DrmsNR")
    ScaleFactor = 1;

  map<pair<pair<TString, int>, TString>, Geometry> mappoints;  // pair = (structure, layer), geometry

  std::sort(filenames.begin(), filenames.end());  //order the files in alphabetical order
  for (TString filename : filenames) {
    int runN;
    TString runstring(filename(regexp));
    if (runstring.IsFloat()) {
      runN = runstring.Atoi();
    } else {
      cout << "ERROR: run number not retrieved for file " << filename << endl;
      continue;
    }

    TFile *f = new TFile(filename, "READ");

    for (TString &structure : structures) {
      TString structname = structure;
      structname.ReplaceAll("_y", "");
      size_t layersnumber = nlayers.at(structname);
      for (size_t layer = 0; layer <= layersnumber; layer++) {
        for (string geometry : geometries) {
          TString name = Variable + "_" + getName(structure, layer, geometry);
          TH1F *histo = dynamic_cast<TH1F *>(f->Get(name));
          //Geometry *geom =nullptr;
          Point *point = nullptr;
          // Three possibilities:
          //  - All histograms are produced correctly
          //  - Only the non-split histograms are produced
          //  - No histogram is produced correctly
          //  FORCE means that the Point is not added to the points collection in the chosen geometry for that structure
          //  If FORCE is not enabled a default value for the Point is used (-9999) which will appear in the plots
          if (!histo) {
            cout << "Run" << runN << " Histogram: " << name << " not found" << endl;
            if (FORCE)
              continue;
            point = new Point(runN, ScaleFactor);
          } else if (structure != "TID" && structure != "TEC") {
            TH1F *histoplus = dynamic_cast<TH1F *>(f->Get((name + "_plus")));
            TH1F *histominus = dynamic_cast<TH1F *>(f->Get((name + "_minus")));
            if (!histoplus || !histominus) {
              cout << "Run" << runN << " Histogram: " << name << " plus or minus not found" << endl;
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
  TString outname = myValidation + "DMRtrends";
  for (TString label : labels) {
    outname += "_";
    outname += label;
  }
  outname += ".root";
  cout << outname << endl;
  TFile *fout = TFile::Open(outname, "RECREATE");
  for (TString &structure : structures) {
    TString structname = structure;
    structname.ReplaceAll("_y", "");
    size_t layersnumber = nlayers.at(structname);
    for (size_t layer = 0; layer <= layersnumber; layer++) {
      for (string geometry : geometries) {
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
          TGraphErrors *g = new TGraphErrors(n, runs.data(), (geom.*trend)().data(), emptyvec.data(), emptyvec.data());
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
          TGraphErrors *g = new TGraphErrors(
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
}

/*! \fn PixelUpdateLines
 *  \brief  Adds to the canvas vertical lines corresponding to the pixelupdateruns
 */
void PixelUpdateLines(TCanvas *c, TString Year, bool showlumi, vector<int> pixelupdateruns) {
  vector<TPaveText *> labels;
  double lastlumi = 0.;
  c->cd();
  size_t index = 0;
  for (int pixelupdaterun : pixelupdateruns) {
    double lumi = 0.;
    if (showlumi)
      lumi = getintegratedlumiuptorun(
          pixelupdaterun,
          Year);  //The vertical line needs to be drawn at the beginning of the run where the pixel update was implemented, thus only the integrated luminosity up to that run is required.
    else
      lumi = pixelupdaterun;
    TLine *line = new TLine(lumi, c->GetUymin(), lumi, c->GetUymax());
    line->SetLineColor(kBlue);
    line->SetLineStyle(9);
    line->Draw();
    //Due to the way the coordinates within the Canvas are set, the following steps are required to draw the TPaveText:
    // Compute the gPad coordinates in TRUE normalized space (NDC)
    int ix1;
    int ix2;
    int iw = gPad->GetWw();
    int ih = gPad->GetWh();
    double x1p, y1p, x2p, y2p;
    gPad->GetPadPar(x1p, y1p, x2p, y2p);
    ix1 = (Int_t)(iw * x1p);
    ix2 = (Int_t)(iw * x2p);
    double wndc = TMath::Min(1., (double)iw / (double)ih);
    double rw = wndc / (double)iw;
    double x1ndc = (double)ix1 * rw;
    double x2ndc = (double)ix2 * rw;
    // Ratios to convert user space in TRUE normalized space (NDC)
    double rx1, ry1, rx2, ry2;
    gPad->GetRange(rx1, ry1, rx2, ry2);
    double rx = (x2ndc - x1ndc) / (rx2 - rx1);
    double _sx;
    // Left limit of the TPaveText
    _sx = rx * (lumi - rx1) + x1ndc;
    // To avoid an overlap between the TPaveText a vertical shift is done when the IOVs are too close
    if (_sx < lastlumi) {
      index++;
    } else
      index = 0;
    TPaveText *box = new TPaveText(_sx + 0.0015, 0.86 - index * 0.04, _sx + 0.035, 0.89 - index * 0.04, "blNDC");
    box->SetFillColor(10);
    box->SetBorderSize(1);
    box->SetLineColor(kBlack);
    TText *textRun = box->AddText(Form("%i", int(pixelupdaterun)));
    textRun->SetTextSize(0.025);
    labels.push_back(box);
    lastlumi = _sx + 0.035;

    gPad->RedrawAxis();
  }
  //Drawing in a separate loop to ensure that the labels are drawn on top of the lines
  for (auto label : labels) {
    label->Draw("same");
  }
  c->Update();
}

/*! \fn getintegratedlumiuptorun
 *  \brief Returns the integrated luminosity up to the run of interest
 */

double getintegratedlumiuptorun(int run, TString Year, double min) {
  TGraph *scale = new TGraph((lumifileperyear(Year, "run")).Data());
  int Nscale = scale->GetN();
  double *xscale = scale->GetX();
  double *yscale = scale->GetY();

  double lumi = min;
  int index = -1;
  for (int j = 0; j < Nscale; j++) {
    lumi += yscale[j];
    if (run >= xscale[j]) {
      index = j;
      continue;
    }
  }
  lumi = min;
  for (int j = 0; j < index; j++)
    lumi += yscale[j] / lumiFactor;

  return lumi;
}
/*! \fn scalebylumi
 *  \brief Scale X-axis of the TGraph and the error on that axis according to the integrated luminosity.
 */

void scalebylumi(TGraphErrors *g, vector<pair<int, double>> lumiIOVpairs) {
  size_t N = g->GetN();
  vector<double> x, y, xerr, yerr;

  //TGraph * scale = new TGraph((lumifileperyear(Year,"IOV")).Data());
  size_t Nscale = lumiIOVpairs.size();

  size_t i = 0;
  while (i < N) {
    double run, yvalue;
    g->GetPoint(i, run, yvalue);
    size_t index = -1;
    for (size_t j = 0; j < Nscale; j++) {
      if (run == (lumiIOVpairs.at(j)
                      .first)) {  //If the starting run of an IOV is included in the list of IOVs, the index is stored
        index = j;
        continue;
      } else if (run > (lumiIOVpairs.at(j).first))
        continue;
    }
    if (lumiIOVpairs.at(index).second == 0 || index < 0.) {
      N = N - 1;
      g->RemovePoint(i);
    } else {
      double xvalue = 0.;
      for (size_t j = 0; j < index; j++)
        xvalue += lumiIOVpairs.at(j).second / lumiFactor;
      x.push_back(xvalue + (lumiIOVpairs.at(index).second / (lumiFactor * 2.)));
      if (yvalue <= DUMMY) {
        y.push_back(DUMMY);
        yerr.push_back(0.);
      } else {
        y.push_back(yvalue);
        yerr.push_back(g->GetErrorY(i));
      }
      xerr.push_back(lumiIOVpairs.at(index).second / (lumiFactor * 2.));
      i = i + 1;
    }
  }
  g->GetHistogram()->Delete();
  g->SetHistogram(nullptr);
  for (size_t i = 0; i < N; i++) {
    g->SetPoint(i, x.at(i), y.at(i));
    g->SetPointError(i, xerr.at(i), yerr.at(i));
  }
}

/*! \fn lumiperIOV
 *  \brief Retrieve luminosity per IOV
 */

vector<pair<int, double>> lumiperIOV(vector<int> IOVlist, TString Year) {
  size_t N = IOVlist.size();
  vector<pair<int, double>> lumiperIOV;
  TGraph *scale = new TGraph((lumifileperyear(Year, "run")).Data());
  size_t Nscale = scale->GetN();
  double *xscale = scale->GetX();
  double *yscale = scale->GetY();

  size_t i = 0;
  size_t index = 0;
  while (i <= N) {
    double run = 0;
    double lumi = 0.;
    if (i != N)
      run = IOVlist.at(i);
    else
      run = 0;
    for (size_t j = index; j < Nscale; j++) {
      if (run == xscale[j]) {
        index = j;
        break;
      } else
        lumi += yscale[j];
    }
    if (i == 0)
      lumiperIOV.push_back(make_pair(0, lumi));
    else
      lumiperIOV.push_back(make_pair(IOVlist.at(i - 1), lumi));
    ++i;
  }
  //for debugging:
  double lumiInput = 0;
  double lumiOutput = 0.;
  for (size_t j = 0; j < Nscale; j++)
    lumiInput += yscale[j];
  //cout << "Total lumi: " << lumiInput <<endl;
  for (size_t j = 0; j < lumiperIOV.size(); j++)
    lumiOutput += lumiperIOV.at(j).second;
  //cout << "Total lumi saved for IOVs: " << lumiOutput <<endl;
  if (abs(lumiInput - lumiOutput) > 0.5) {
    cout << "ERROR: luminosity retrieved for IOVs does not match the one for the runs" << endl
         << "Please check that all IOV first runs are part of the run-per-lumi file!" << endl;
    exit(EXIT_FAILURE);
  }
  return lumiperIOV;
}

/*! \fn ConvertToHist
 *  \brief A TH1F is constructed using the points and the errors collected in the TGraphErrors
 */

TH1F *ConvertToHist(TGraphErrors *g) {
  size_t N = g->GetN();
  double *x = g->GetX();
  double *y = g->GetY();
  double *xerr = g->GetEX();
  vector<float> bins;
  bins.push_back(x[0] - xerr[0]);
  for (size_t i = 1; i < N; i++) {
    if ((x[i - 1] + xerr[i - 1]) > (bins.back() + pow(10, -6)))
      bins.push_back(x[i - 1] + xerr[i - 1]);
    if ((x[i] - xerr[i]) > (bins.back() + pow(10, -6)))
      bins.push_back(x[i] - xerr[i]);
  }
  bins.push_back(x[N - 1] + xerr[N - 1]);
  TString histoname = "histo_";
  histoname += g->GetName();
  TH1F *histo = new TH1F(histoname, g->GetTitle(), bins.size() - 1, bins.data());
  for (size_t i = 0; i < N; i++) {
    histo->Fill(x[i], y[i]);
    histo->SetBinError(histo->FindBin(x[i]), pow(10, -6));
  }
  return histo;
}

/*! \fn PlotDMRTrends
 *  \brief Plot the DMR trends.
 */

void PlotDMRTrends(vector<int> IOVlist,
                   TString Variable,
                   vector<string> labels,
                   TString Year,
                   string myValidation,
                   vector<string> geometries,
                   vector<Color_t> colours,
                   TString outputdir,
                   bool pixelupdate,
                   vector<int> pixelupdateruns,
                   bool showlumi) {
  gErrorIgnoreLevel = kWarning;
  checkrunlist(pixelupdateruns, {}, Year);
  vector<TString> structures{"BPIX", "BPIX_y", "FPIX", "FPIX_y", "TIB", "TID", "TOB", "TEC"};

  const map<TString, int> nlayers = numberOfLayers(Year);

  vector<pair<int, double>> lumiIOVpairs;
  if (showlumi)
    lumiIOVpairs = lumiperIOV(IOVlist, Year);

  TString filename = myValidation + "DMRtrends";
  for (TString label : labels) {
    filename += "_";
    filename += label;
  }
  filename += ".root";
  cout << filename << endl;
  TFile *in = new TFile(filename);
  for (TString &structure : structures) {
    TString structname = structure;
    structname.ReplaceAll("_y", "");
    int layersnumber = nlayers.at(structname);
    for (int layer = 0; layer <= layersnumber; layer++) {
      vector<TString> variables{"mu",
                                "sigma",
                                "muplus",
                                "sigmaplus",
                                "muminus",
                                "sigmaminus",
                                "deltamu",
                                "sigmadeltamu",
                                "musigma",
                                "muplussigmaplus",
                                "muminussigmaminus",
                                "deltamusigmadeltamu"};
      vector<string> YaxisNames{
          "#mu [#mum]",
          "#sigma_{#mu} [#mum]",
          "#mu outward [#mum]",
          "#sigma_{#mu outward} [#mum]",
          "#mu inward [#mum]",
          "#sigma_{#mu inward} [#mum]",
          "#Delta#mu [#mum]",
          "#sigma_{#Delta#mu} [#mum]",
          "#mu [#mum]",
          "#mu outward [#mum]",
          "#mu inward [#mum]",
          "#Delta#mu [#mum]",
      };
      if (Variable == "DrmsNR")
        YaxisNames = {
            "RMS(x'_{pred}-x'_{hit} /#sigma)",
            "#sigma_{RMS(x'_{pred}-x'_{hit} /#sigma)}",
            "RMS(x'_{pred}-x'_{hit} /#sigma) outward",
            "#sigma_{#mu outward}",
            "RMS(x'_{pred}-x'_{hit} /#sigma) inward",
            "#sigma_{RMS(x'_{pred}-x'_{hit} /#sigma) inward}",
            "#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)",
            "#sigma_{#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)}",
            "RMS(x'_{pred}-x'_{hit} /#sigma)",
            "RMS(x'_{pred}-x'_{hit} /#sigma) outward",
            "RMS(x'_{pred}-x'_{hit} /#sigma) inward",
            "#DeltaRMS(x'_{pred}-x'_{hit} /#sigma)",
        };
      //For debugging purposes we still might want to have a look at plots for a variable without errors, once ready for the PR those variables will be removed and the iterator will start from 0
      for (size_t i = 0; i < variables.size(); i++) {
        TString variable = variables.at(i);
        if (variable.Contains("plus") || variable.Contains("minus") || variable.Contains("delta")) {
          if (structname == "TEC" || structname == "TID")
            continue;  //Lorentz drift cannot appear in TEC and TID. These structures are skipped when looking at outward and inward pointing modules.
        }
        TCanvas *c = new TCanvas("dummy", "", 2000, 800);

        vector<Color_t>::iterator colour = colours.begin();

        TMultiGraph *mg = new TMultiGraph(structure, structure);
        THStack *mh = new THStack(structure, structure);
        size_t igeom = 0;
        for (string geometry : geometries) {
          TString name = Variable + "_" + getName(structure, layer, geometry);
          TGraphErrors *g = dynamic_cast<TGraphErrors *>(in->Get(name + "_" + variables.at(i)));
          g->SetName(name + "_" + variables.at(i));
          if (i >= 8) {
            g->SetLineWidth(1);
            g->SetLineColor(*colour);
            g->SetFillColorAlpha(*colour, 0.2);
          }
          vector<vector<double>> vectors;
          //if(showlumi&&i<8)scalebylumi(dynamic_cast<TGraph*>(g));
          if (showlumi)
            scalebylumi(g, lumiIOVpairs);
          g->SetLineColor(*colour);
          g->SetMarkerColor(*colour);
          TH1F *h = ConvertToHist(g);
          h->SetLineColor(*colour);
          h->SetMarkerColor(*colour);
          h->SetMarkerSize(0);
          h->SetLineWidth(1);

          if (i < 8) {
            mg->Add(g, "PL");
            mh->Add(h, "E");
          } else {
            mg->Add(g, "2");
            mh->Add(h, "E");
          }
          ++colour;
          ++igeom;
        }

        gStyle->SetOptTitle(0);
        gStyle->SetPadLeftMargin(0.08);
        gStyle->SetPadRightMargin(0.05);
        gPad->SetTickx();
        gPad->SetTicky();
        gStyle->SetLegendTextSize(0.025);

        double max = 6;
        double min = -4;
        if (Variable == "DrmsNR") {
          if (variable.Contains("delta")) {
            max = 0.15;
            min = -0.1;
          } else {
            max = 1.2;
            min = 0.6;
          }
        }
        double range = max - min;

        if (((variable == "sigma" || variable == "sigmaplus" || variable == "sigmaminus" ||
              variable == "sigmadeltamu") &&
             range >= 2)) {
          mg->SetMaximum(4);
          mg->SetMinimum(-2);
        } else {
          mg->SetMaximum(max + range * 0.1);
          mg->SetMinimum(min - range * 0.3);
        }

        if (i < 8) {
          mg->Draw("a");
        } else {
          mg->Draw("a2");
        }

        char *Ytitle = (char *)YaxisNames.at(i).c_str();
        mg->GetYaxis()->SetTitle(Ytitle);
        mg->GetXaxis()->SetTitle(showlumi ? "Integrated lumi [1/fb]" : "IOV number");
        mg->GetXaxis()->CenterTitle(true);
        mg->GetYaxis()->CenterTitle(true);
        mg->GetYaxis()->SetTitleOffset(.5);
        mg->GetYaxis()->SetTitleSize(.05);
        mg->GetXaxis()->SetTitleSize(.04);
        if (showlumi)
          mg->GetXaxis()->SetLimits(0., mg->GetXaxis()->GetXmax());

        c->Update();

        TLegend *legend = c->BuildLegend();
        // TLegend *legend = c->BuildLegend(0.15,0.18,0.15,0.18);
        int Ngeom = geometries.size();
        if (Ngeom >= 6)
          legend->SetNColumns(2);
        else if (Ngeom >= 9)
          legend->SetNColumns(3);
        else
          legend->SetNColumns(1);
        TString structtitle = "#bf{";
        if (structure.Contains("PIX") && !(structure.Contains("_y")))
          structtitle += structure + " (x)";
        else if (structure.Contains("_y")) {
          TString substring(structure(0, 4));
          structtitle += substring + " (y)";
        } else
          structtitle += structure;
        if (layer != 0) {
          if (structure == "TID" || structure == "TEC" || structure == "FPIX" || structure == "FPIX_y")
            structtitle += "  disc ";
          else
            structtitle += "  layer ";
          structtitle += layer;
        }
        structtitle += "}";
        PixelUpdateLines(c, Year, showlumi, pixelupdateruns);

        TPaveText *CMSworkInProgress = new TPaveText(
            0, mg->GetYaxis()->GetXmax() + range * 0.02, 2.5, mg->GetYaxis()->GetXmax() + range * 0.12, "nb");
        CMSworkInProgress->AddText("#scale[1.1]{CMS} #bf{Internal}");
        CMSworkInProgress->SetTextAlign(12);
        CMSworkInProgress->SetTextSize(0.04);
        CMSworkInProgress->SetFillColor(10);
        CMSworkInProgress->Draw();
        TPaveText *TopRightCorner = new TPaveText(0.95 * (mg->GetXaxis()->GetXmax()),
                                                  mg->GetYaxis()->GetXmax() + range * 0.02,
                                                  (mg->GetXaxis()->GetXmax()),
                                                  mg->GetYaxis()->GetXmax() + range * 0.12,
                                                  "nb");
        TopRightCorner->AddText(Year + " pp collisions");
        TopRightCorner->SetTextAlign(32);
        TopRightCorner->SetTextSize(0.04);
        TopRightCorner->SetFillColor(10);
        TopRightCorner->Draw();
        TPaveText *structlabel = new TPaveText(0.95 * (mg->GetXaxis()->GetXmax()),
                                               mg->GetYaxis()->GetXmin() + range * 0.02,
                                               0.99 * (mg->GetXaxis()->GetXmax()),
                                               mg->GetYaxis()->GetXmin() + range * 0.12,
                                               "nb");
        structlabel->AddText(structtitle.Data());
        structlabel->SetTextAlign(32);
        structlabel->SetTextSize(0.04);
        structlabel->SetFillColor(10);
        structlabel->Draw();

        legend->Draw();
        mh->Draw("nostack same");
        c->Update();
        TString structandlayer = getName(structure, layer, "");
        TString printfile = outputdir;
        if (!(outputdir.EndsWith("/")))
          outputdir += "/";
        for (TString label : labels) {
          printfile += label;
          printfile += "_";
        }
        printfile += Variable;
        printfile += "_";
        printfile += variable + structandlayer;
        c->SaveAs(printfile + ".pdf");
        c->SaveAs(printfile + ".eps");
        c->SaveAs(printfile + ".png");
        c->Destructor();
      }
    }
  }
  in->Close();
}

/*! \fn main
 *  \brief main function: if no arguments are specified a default list of arguments is used, otherwise a total of 9 arguments are required:
 * @param IOVlist:                 string containing the list of IOVs separated by a ","
 * @param labels:                  string containing labels that must be part of the input files
 * @param Year:                    string containing the year of the studied runs (needed to retrieve the lumi-per-run file)
 * @param pathtoDMRs:              string containing the path to the directory where the DMRs are stored
 * @param geometrieandcolours:     string containing the list of geometries and colors in the following way name1:color1,name2:color2 etc.
 * @param outputdirectory:         string containing the output directory for the plots
 * @param pixelupdatelist:         string containing the list of pixelupdates separated by a ","
 * @param showpixelupdate:         boolean that if set to true will allow to plot vertical lines in the canvas corresponding to the pixel updates
 * @param showlumi:                boolean, if set to false the trends will be presented in function of the run (IOV) number, if set to true the luminosity is used on the x axis
 * @param FORCE:              //!< boolean, if set to true the plots will be made regardless of possible errors.
 *                                 Eventual errors while running the code will be ignored and just warnings will appear in the output.
 */

int main(int argc, char *argv[]) {
  if (argc == 1) {
    vector<int> IOVlist = {314881, 315257, 315488, 315489, 315506, 316239, 316271, 316361, 316363, 316378, 316456,
                           316470, 316505, 316569, 316665, 316758, 317080, 317182, 317212, 317295, 317339, 317382,
                           317438, 317527, 317661, 317664, 318712, 319337, 319460, 320841, 320854, 320856, 320888,
                           320916, 320933, 320980, 321009, 321119, 321134, 321164, 321261, 321294, 321310, 321393,
                           321397, 321431, 321461, 321710, 321735, 321773, 321774, 321778, 321820, 321831, 321880,
                           321960, 322014, 322510, 322603, 323232, 323423, 323472, 323475, 323693, 323794, 323976,
                           324202, 324206, 324245, 324729, 324764, 324840, 324999, 325097, 325110};
    vector<int> pixelupdateruns{316758, 317527, 317661, 317664, 318227, 320377};  //2018
    //	vector<int> pixelupdateruns {290543, 297281, 298653, 299443, 300389, 301046, 302131, 303790, 303998, 304911, 305898};//2017

    cout << "WARNING: Running function with arguments specified in DMRtrends.cc" << endl
         << "If you want to specify the arguments from command line run the macro as follows:" << endl
         << "DMRtrends labels pathtoDMRs geometriesandcolourspairs outputdirectory showpixelupdate showlumi FORCE"
         << endl;

    //PLEASE READ: for debugging purposes please keep at least one example that works commented.
    //             Error messages are still a W.I.P. and having a working example available is useful for debugging.
    //Example provided for a currently working set of parameters:
    DMRtrends(IOVlist,
              {"median", "DrmsNR"},
              {"minbias"},
              "2018",
              "/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/data/commonValidation/results/adewit/UL18_DMR_MB_eoj_v3/",
              {"1-step hybrid", "mid18 rereco", "counter twist"},
              {kBlue, kBlack, kGreen + 3},
              "/afs/cern.ch/user/a/acardini/commonValidation/results/acardini/DMRs/DMRTrends/test/",
              true,
              pixelupdateruns,
              true,
              true);
    return 0;
  } else if (argc < 12) {
    cout << "DMRtrends IOVlist labels Year pathtoDMRs geometriesandcolourspairs outputdirectory pixelupdatelist "
            "showpixelupdate showlumi FORCE"
         << endl;

    return 1;
  }

  TString runlist = argv[1], all_variables = argv[2], all_labels = argv[3], Year = argv[4], pathtoDMRs = argv[5],
          geometrieandcolours = argv[6],  //name1:title1:color1,name2:title2:color2,name3:title3:color3
      outputdirectory = argv[7], pixelupdatelist = argv[8];
  bool showpixelupdate = argv[9], showlumi = argv[10], FORCE = argv[11];
  TObjArray *vararray = all_variables.Tokenize(",");
  vector<string> Variables;
  for (int i = 0; i < vararray->GetEntries(); i++)
    Variables.push_back((string)(vararray->At(i)->GetName()));
  TObjArray *labelarray = all_labels.Tokenize(",");
  vector<string> labels;
  for (int i = 0; i < labelarray->GetEntries(); i++)
    labels.push_back((string)(labelarray->At(i)->GetName()));
  TObjArray *IOVarray = runlist.Tokenize(",");
  vector<int> IOVlist;
  for (int i = 0; i < IOVarray->GetEntries(); i++)
    IOVlist.push_back(stoi(IOVarray->At(i)->GetName()));
  vector<int> pixelupdateruns;
  TObjArray *PIXarray = pixelupdatelist.Tokenize(",");
  for (int i = 0; i < PIXarray->GetEntries(); i++)
    pixelupdateruns.push_back(stoi(PIXarray->At(i)->GetName()));
  vector<string> geometries;
  //TO DO: the color is not taken correctly from command line
  vector<Color_t> colours;
  TObjArray *geometrieandcolourspairs = geometrieandcolours.Tokenize(",");
  for (int i = 0; i < geometrieandcolourspairs->GetEntries(); i++) {
    TObjArray *geomandcolourvec = TString(geometrieandcolourspairs->At(i)->GetName()).Tokenize(":");
    geometries.push_back(geomandcolourvec->At(0)->GetName());
    colours.push_back(ColorParser(geomandcolourvec->At(1)->GetName()));
  }
  DMRtrends(IOVlist,
            Variables,
            labels,
            Year,
            pathtoDMRs.Data(),
            geometries,
            colours,
            outputdirectory.Data(),
            showpixelupdate,
            pixelupdateruns,
            showlumi,
            FORCE);

  return 0;
}
