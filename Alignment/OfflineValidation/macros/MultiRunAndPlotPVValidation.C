#include "TArrow.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TList.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TProcPool.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TSystemFile.h"
#include <TStopwatch.h>
#include <algorithm>
#include <bitset>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

/*!
 * \def boolean to decide if it is in debug mode
 */
#define DEBUG true

/*!
 * \def number of workers
 */
const size_t nWorkers = 10;

/*!
 * \def basically the y-values of a TGraph
 */
typedef std::map<TString, std::vector<double> > alignmentTrend;

namespace pv {
  enum view { dxyphi, dzphi, dxyeta, dzeta, pT, generic };

  /*! \fn closest
   *  \brief method to find first value that doesn not compare left
   */

  int closest(std::vector<int> const &vec, int value) {
    auto const it = std::lower_bound(vec.begin(), vec.end(), value);
    if (it == vec.end()) {
      return -1;
    }
    return *it;
  }

  const Int_t markers[8] = {kFullSquare,
                            kFullCircle,
                            kFullTriangleDown,
                            kOpenSquare,
                            kOpenCircle,
                            kFullTriangleUp,
                            kOpenTriangleDown,
                            kOpenTriangleUp};
  const Int_t colors[8] = {kBlack, kRed, kBlue, kGreen + 2, kMagenta, kViolet, kCyan, kYellow};

  /*! \struct biases
   *  \brief Structure biases
   *         Contains characterization of a single run PV bias plot
   *
   * @param m_mean:             mean value of the profile points
   * @param m_rms:              RMS value of the profle points
   * @param m_w_mean:           mean weighted on the errors
   * @param m_w_rms:            RMS weighted on the errors
   * @param m_min:              minimum of the profile
   * @param m_max:              maximum of the profile
   * @param m_chi2:             chi2 of a liner fit
   * @param m_ndf:              number of the dof of the linear fit
   * @param m_ks:               Kolmogorov-Smirnov score of comparison with flat line
   */

  struct biases {
    // contructor
    biases(double mean, double rms, double wmean, double wrms, double min, double max, double chi2, int ndf, double ks) {
      m_mean = mean;
      m_rms = rms;
      m_w_mean = wmean;
      m_w_rms = wrms;
      m_min = min;
      m_max = max;
      m_chi2 = chi2;
      m_ndf = ndf;
      m_ks = ks;
    }

    // empty constructor
    biases() { init(); }

    /*! \fn init
     *  \brief initialising all members one by one
     */

    void init() {
      m_mean = 0;
      m_rms = 0.;
      m_min = +999.;
      m_max = -999.;
      m_w_mean = 0.;
      m_w_rms = 0.;
      m_chi2 = -1.;
      m_ndf = 0.;
      m_ks = 9999.;
    }

    double getMean() { return m_mean; }
    double getWeightedMean() { return m_w_mean; }
    double getRMS() { return m_rms; }
    double getWeightedRMS() { return m_w_rms; }
    double getMin() { return m_min; }
    double getMax() { return m_max; }
    double getChi2() { return m_chi2; }
    double getNDF() { return m_ndf; }
    double getNormChi2() { return double(m_chi2) / double(m_ndf); }
    double getChi2Prob() { return TMath::Prob(m_chi2, m_ndf); }
    double getKSScore() { return m_ks; }

  private:
    double m_mean;
    double m_min;
    double m_max;
    double m_rms;
    double m_w_mean;
    double m_w_rms;
    double m_chi2;
    int m_ndf;
    double m_ks;
  };

  /*! \struct wrappedTrends
   *  \brief Structure wrappedTrends
   *         Contains the ensemble vs run number of the alignmentTrend characterization
   *
   * @param mean:             alignmentTrend of the mean value of the profile points
   * @param low:              alignmentTrend of the lowest value of the profle points
   * @param high:             alignmentTrend of the highest value of the profile points
   * @param lowerr:           alignmentTrend of the difference between the lowest value and the mean of the profile points
   * @param higherr:          alignmentTrend of the difference between the highest value and the mean of the profile points
   * @param chi2:             alignmentTrend of the chi2 value of a linear fit to the profile points
   * @param KS:               alignmentTrend of the Kolmogorow-Smirnov score of the comarison of the profile points to a flat line
   */

  struct wrappedTrends {
    /*! \fn wrappedTrends
     *  \brief Constructor of structure wrappedTrends, initialising all members from DMRs directly (with split)
     */

    wrappedTrends(alignmentTrend mean,
                  alignmentTrend low,
                  alignmentTrend high,
                  alignmentTrend lowerr,
                  alignmentTrend higherr,
                  alignmentTrend chi2,
                  alignmentTrend KS) {
      std::cout << "pv::wrappedTrends c'tor" << std::endl;

      m_mean = mean;
      m_low = low;
      m_high = high;
      m_lowerr = lowerr;
      m_higherr = higherr;
      m_chi2 = chi2;
      m_KS = KS;
    }

    alignmentTrend getMean() const { return m_mean; }
    alignmentTrend getLow() const { return m_low; }
    alignmentTrend getHigh() const { return m_high; }
    alignmentTrend getLowErr() const { return m_lowerr; }
    alignmentTrend getHighErr() const { return m_higherr; }
    alignmentTrend getChi2() const { return m_chi2; }
    alignmentTrend getKS() const { return m_KS; }

  private:
    alignmentTrend m_mean;
    alignmentTrend m_low;
    alignmentTrend m_high;
    alignmentTrend m_lowerr;
    alignmentTrend m_higherr;
    alignmentTrend m_chi2;
    alignmentTrend m_KS;
  };

  /*! \struct bundle
   *  \brief Structure bundle
   *         Contains the ensemble of all the information to build the graphs alignmentTrends
   *
   * @param nObjects                     int, number of alignments to be considered
   * @param dataType                     TString, type of the data to be displayed (time, lumi)
   * @param dataTypeLabel                TString, x-axis label
   * @param lumiMapByrun                 std::map of the luminoisty by run number
   * @param times                        std::map of the date (UTC) by run number
   * @param lumiaxisformat               boolean, is the x-axis of the type lumi
   * @param timeaxisformat               boolean, is the x-axis of the type time
   */

  struct bundle {
    bundle(int nObjects,
           const TString &dataType,
           const TString &dataTypeLabel,
           const std::map<int, double> &lumiMapByRun,
           const std::map<int, TDatime> &times,
           const bool &lumiaxisformat,
           const bool &timeaxisformat,
           const bool &useRMS) {
      m_nObjects = nObjects;
      m_datatype = dataType.Data();
      m_datatypelabel = dataTypeLabel.Data();
      m_lumiMapByRun = lumiMapByRun;
      m_times = times;
      m_useRMS = useRMS;

      std::cout << "pv::bundle c'tor: " << dataTypeLabel << " member: " << m_datatypelabel << std::endl;

      // make sure you don't use them at the same time
      if (lumiaxisformat || timeaxisformat) {
        assert(lumiaxisformat != timeaxisformat);
      }

      if (lumiaxisformat) {
        std::cout << "is lumiaxis format" << std::endl;
        m_axis_types.set(0);
      } else if (timeaxisformat) {
        std::cout << "is timeaxis format" << std::endl;
        m_axis_types.set(1);
      } else {
        std::cout << "is runaxis format" << std::endl;
      }

      std::cout << m_axis_types << std::endl;

      m_totalLumi = lumiMapByRun.rbegin()->second;
    }

    int getNObjects() const { return m_nObjects; }
    const char *getDataType() const { return m_datatype; }
    const char *getDataTypeLabel() const { return m_datatypelabel; }
    const std::map<int, double> getLumiMapByRun() const { return m_lumiMapByRun; }
    double getTotalLumi() const { return m_totalLumi; }
    const std::map<int, TDatime> getTimes() const { return m_times; }
    bool isLumiAxis() const { return m_axis_types.test(0); }
    bool isTimeAxis() const { return m_axis_types.test(1); }
    bool isUsingRMS() const { return m_useRMS; }
    void printAll() {
      std::cout << "dataType      " << m_datatype << std::endl;
      std::cout << "dataTypeLabel " << m_datatypelabel << std::endl;
      if (this->isLumiAxis())
        std::cout << "is lumi axis" << std::endl;
      if (this->isTimeAxis())
        std::cout << "is time axis" << std::endl;
    }

  private:
    int m_nObjects;
    const char *m_datatype;
    const char *m_datatypelabel;
    float m_totalLumi;
    std::map<int, double> m_lumiMapByRun;
    std::map<int, TDatime> m_times;
    std::bitset<2> m_axis_types;
    bool m_useRMS;
  };

}  // namespace pv

// auxilliary struct to store
// histogram features
struct unrolledHisto {
  double m_y_min;
  double m_y_max;
  unsigned int m_n_bins;
  std::vector<double> m_bincontents;

  unrolledHisto() {
    m_y_min = 0.;
    m_y_max = 0.;
    m_n_bins = 0;
    m_bincontents.clear();
  }  // first constructor empty

  unrolledHisto(const double &y_min,
                const double &y_max,
                const unsigned int &n_bins,
                const std::vector<double> &bincontents) {
    m_y_min = y_min;
    m_y_max = y_max;
    m_n_bins = n_bins;
    m_bincontents = bincontents;
  }  //look, a constructor

  double get_y_min() { return m_y_min; }

  double get_y_max() { return m_y_max; }

  unsigned int get_n_bins() { return m_n_bins; }

  std::vector<double> get_bin_contents() { return m_bincontents; }

  double get_integral() {
    double ret(0.);
    for (const auto &binc : m_bincontents) {
      ret += binc;
    }
    return ret;
  }
};

/*! \struct outTrends
   *  \brief Structure outTrends
   *         Contains the ensemble of all the alignmentTrends built by the functor
   *
   * @param m_index                     int, to keep track of which chunk of data has been processed
   * @param m_lumiSoFar                 double, luminosity in this section of the data
   * @param m_runs                      std::vector, list of the run processed in this section
   * @param m_lumiByRun                 std::vector, list of the luminisoties per run, indexed
   * @param m_lumiMarpByRun             std::map, map of the luminosities per run
   * @param m_dxyPhiMeans               alignmentTrend of the mean values of the profile dxy vs phi
   * @param m_dxyPhiChi2                alignmentTrend of chi2 of the linear fit per profile dxy vs phi
   * @param m_dxyPhiKS                  alignmentTrend of Kolmogorow-Smirnov score of comparison of dxy vs phi profile with flat line
   * @param m_dxyPhiHi                  alignmentTrend of the highest value of the profile dxy vs phi
   * @param m_dxyPhiLo                  alignmentTrend of the lowest value of the profile dxy vs phi
   * @param m_dxyEtaMeans               alignmentTrend of the mean values of the profile dxy vs eta
   * @param m_dxyEtaChi2                alignmentTrend of chi2 of the linear fit per profile dxy vs eta
   * @param m_dxyEtaKS                  alignmentTrend of Kolmogorow-Smirnov score of comparison of dxy vs eta profile with flat line
   * @param m_dxyEtaHi                  alignmentTrend of the highest value of the profile dxy vs eta
   * @param m_dxyEtaLo                  alignmentTrend of the lowest value of the profile dxy vs eta
   * @param m_dzPhiMeans                alignmentTrend of the mean values of the profile dz vs phi
   * @param m_dzPhiChi2                 alignmentTrend of chi2 of the linear fit per profile dz vs phi
   * @param m_dzPhiKS                   alignmentTrend of Kolmogorow-Smirnov score of comparison of dz vs phi profile with flat line
   * @param m_dzPhiHi                   alignmentTrend of the highest value of the profile dz vs phi
   * @param m_dzPhiLo                   alignmentTrend of the lowest value of the profile dz vs phi
   * @param m_dzEtaMeans                alignmentTrend of the mean values of the profile dz vs eta
   * @param m_dzEtaChi2                 alignmentTrend of chi2 of the linear fit per profile dz vs eta
   * @param m_dzEtaKS                   alignmentTrend of Kolmogorow-Smirnov score of comparison of dz vs eta profile with flat line
   * @param m_dzEtaHi                   alignmentTrend of the highest value of the profile dz vs eta
   * @param m_dzEtaLo                   alignmentTrend of the lowest value of the profile dz vs eta
   * @param m_dxyVect                   map of the unrolled histograms for dxy residuals
   * @param m_dzVect                    map of the unrolled histograms for dz residulas
   */

struct outTrends {
  int m_index;
  double m_lumiSoFar;
  std::vector<double> m_runs;
  std::vector<double> m_lumiByRun;
  std::map<int, double> m_lumiMapByRun;
  alignmentTrend m_dxyPhiMeans;
  alignmentTrend m_dxyPhiChi2;
  alignmentTrend m_dxyPhiKS;
  alignmentTrend m_dxyPhiHi;
  alignmentTrend m_dxyPhiLo;
  alignmentTrend m_dxyEtaMeans;
  alignmentTrend m_dxyEtaChi2;
  alignmentTrend m_dxyEtaKS;
  alignmentTrend m_dxyEtaHi;
  alignmentTrend m_dxyEtaLo;
  alignmentTrend m_dzPhiMeans;
  alignmentTrend m_dzPhiChi2;
  alignmentTrend m_dzPhiKS;
  alignmentTrend m_dzPhiHi;
  alignmentTrend m_dzPhiLo;
  alignmentTrend m_dzEtaMeans;
  alignmentTrend m_dzEtaChi2;
  alignmentTrend m_dzEtaKS;
  alignmentTrend m_dzEtaHi;
  alignmentTrend m_dzEtaLo;
  std::map<TString, std::vector<unrolledHisto> > m_dxyVect;
  std::map<TString, std::vector<unrolledHisto> > m_dzVect;

  void init() {
    m_index = -1;
    m_lumiSoFar = 0.;
    m_runs.clear();
    m_lumiByRun.clear();
    m_lumiMapByRun.clear();

    m_dxyPhiMeans.clear();
    m_dxyPhiChi2.clear();
    m_dxyPhiKS.clear();
    m_dxyPhiHi.clear();
    m_dxyPhiLo.clear();

    m_dxyEtaMeans.clear();
    m_dxyEtaChi2.clear();
    m_dxyEtaKS.clear();
    m_dxyEtaHi.clear();
    m_dxyEtaLo.clear();

    m_dzPhiMeans.clear();
    m_dzPhiChi2.clear();
    m_dzPhiKS.clear();
    m_dzPhiHi.clear();
    m_dzPhiLo.clear();

    m_dzEtaMeans.clear();
    m_dzEtaChi2.clear();
    m_dzEtaKS.clear();
    m_dzEtaHi.clear();
    m_dzEtaLo.clear();

    m_dxyVect.clear();
    m_dzVect.clear();
  }
};

// forward declarations
void MultiRunPVValidation(TString namesandlabels = "",
                          bool lumi_axis_format = false,
                          bool time_axis_format = false,
                          bool useRMS = true);
outTrends processData(size_t iter,
                      std::vector<int> intersection,
                      const Int_t nDirs_,
                      const char *dirs[10],
                      TString LegLabels[10],
                      bool useRMS);

void arrangeOutCanvas(TCanvas *canv,
                      TH1F *m_11Trend[100],
                      TH1F *m_12Trend[100],
                      TH1F *m_21Trend[100],
                      TH1F *m_22Trend[100],
                      Int_t nFiles,
                      TString LegLabels[10],
                      unsigned int theRun);

pv::biases getBiases(TH1F *hist);
unrolledHisto getUnrolledHisto(TH1F *hist);

TH1F *DrawConstant(TH1F *hist, Int_t iter, Double_t theConst);
TH1F *DrawConstantWithErr(TH1F *hist, Int_t iter, Double_t theConst);
TH1F *DrawConstantGraph(TGraph *graph, Int_t iter, Double_t theConst);
std::vector<int> list_files(const char *dirname = ".", const char *ext = ".root");
TH1F *checkTH1AndReturn(TFile *f, TString address);
void MakeNiceTrendPlotStyle(TH1 *hist, Int_t color);
void cmsPrel(TPad *pad, size_t ipads = 1);
void makeNewXAxis(TH1 *h);
void beautify(TGraph *g);
void beautify(TH1 *h);
void adjustmargins(TCanvas *canv);
void adjustmargins(TVirtualPad *canv);
void setStyle();
pv::view checkTheView(const TString &toCheck);
template <typename T>
void timify(T *mgr);
Double_t getMaximumFromArray(TObjArray *array);
void superImposeIOVBoundaries(TCanvas *c,
                              bool lumi_axis_format,
                              bool time_axis_format,
                              const std::map<int, double> &lumiMapByRun,
                              const std::map<int, TDatime> &timeMap,
                              bool drawText = true);
void outputGraphs(const pv::wrappedTrends &allInputs,
                  const std::vector<double> &ticks,
                  const std::vector<double> &ex_ticks,
                  TCanvas *&canv,
                  TCanvas *&mean_canv,
                  TCanvas *&rms_canv,
                  TGraph *&g_mean,
                  TGraph *&g_chi2,
                  TGraph *&g_KS,
                  TGraph *&g_low,
                  TGraph *&g_high,
                  TGraphAsymmErrors *&g_asym,
                  TH1F *h_RMS[],
                  const pv::bundle &mybundle,
                  const pv::view &theView,
                  const int index,
                  TObjArray *&array,
                  const TString &label,
                  TLegend *&legend);

/*! \fn split
 *  \brief utility function to split strings
 */

/*--------------------------------------------------------------------*/
std::vector<std::string> split(const std::string &s, char delimiter)
/*--------------------------------------------------------------------*/
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

///////////////////////////////////
//
//  Main function
//
///////////////////////////////////

void MultiRunPVValidation(TString namesandlabels, bool lumi_axis_format, bool time_axis_format, bool useRMS) {
  TStopwatch timer;
  timer.Start();

  using namespace std::placeholders;  // for _1, _2, _3...
  gROOT->ProcessLine("gErrorIgnoreLevel = kError;");

  ROOT::EnableThreadSafety();
  TH1::AddDirectory(kFALSE);

  // consistency check, we cannot do plot vs lumi if time_axis
  if (lumi_axis_format && time_axis_format) {
    std::cout << "##########################################################################################"
              << std::endl;
    std::cout << "msg-i: MultiRunPVValidation(): you're requesting both summary vs lumi and vs time, " << std::endl;
    std::cout << "       this combination is inconsistent --> exiting!" << std::endl;
    //return;
    exit(EXIT_FAILURE);
  }

  // preload the dates from file
  std::map<int, TDatime> times;

  if (time_axis_format) {
    std::ifstream infile("times.txt");

    if (!infile) {
      std::cout << "missing input file :(" << std::endl;
      std::cout << " -- exiting" << std::endl;
      return;
    }

    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::string a, b, c;
      if (!(iss >> a >> b >> c)) {
        break;
      }  // error

      //std::cout<<a<<"  "<<b<<"   "<<c<<"   "<<std::endl;

      int run = std::stoi(a);
      auto tokens_b = split(b, '-');
      int year = std::stoi(tokens_b[0]);
      int month = std::stoi(tokens_b[1]);
      int day = std::stoi(tokens_b[2]);

      auto tokens_c = split(c, '.');
      auto tokens_c1 = split(tokens_c[0], ':');

      int hour = std::stoi(tokens_c1[0]);
      int minute = std::stoi(tokens_c1[2]);
      int second = std::stoi(tokens_c1[2]);

      //std::cout<<run<<" "<<year<<" "<<month<<" "<<day<<" "<<hour<<" "<<minute<<" "<<second<<" "<<std::endl;

      TDatime da(year, month, day, hour, minute, second);
      times[run] = da;
    }
  }  // if time axis in the plots

  //std::ofstream outfile ("lumiByRun.txt");
  std::ofstream outfile("log.txt");
  setStyle();

  TList *DirList = new TList();
  TList *LabelList = new TList();

  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
    TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");

    if (aFileLegPair->GetEntries() == 2) {
      DirList->Add(aFileLegPair->At(0));
      LabelList->Add(aFileLegPair->At(1));
    } else {
      std::cout << "Please give file name and legend entry in the following form:\n"
                << " filename1=legendentry1,filename2=legendentry2\n";
    }
  }

  const Int_t nDirs_ = DirList->GetSize();
  TString LegLabels[10];
  const char *dirs[10];

  std::vector<int> intersection;
  std::vector<double> runs;
  std::vector<double> lumiByRun;
  std::map<int, double> lumiMapByRun;
  std::vector<double> x_ticks;
  std::vector<double> ex_ticks = {0.};

  std::vector<double> runtimes;
  if (time_axis_format) {
    for (const auto &element : times) {
      runtimes.push_back((element.second).Convert());
    }
  }

  for (Int_t j = 0; j < nDirs_; j++) {
    // Retrieve labels
    TObjString *legend = (TObjString *)LabelList->At(j);
    TObjString *dir = (TObjString *)DirList->At(j);
    LegLabels[j] = legend->String();
    dirs[j] = (dir->String()).Data();
    cout << "MultiRunPVValidation(): label[" << j << "]" << LegLabels[j] << endl;

    std::vector<int> currentList = list_files(dirs[j]);
    std::vector<int> tempSwap;

    std::sort(currentList.begin(), currentList.end());

    if (j == 0) {
      intersection = currentList;
    }

    std::sort(intersection.begin(), intersection.end());

    std::set_intersection(
        currentList.begin(), currentList.end(), intersection.begin(), intersection.end(), std::back_inserter(tempSwap));

    intersection.clear();
    intersection = tempSwap;
    tempSwap.clear();
  }

  // debug only
  for (UInt_t index = 0; index < intersection.size(); index++) {
    std::cout << index << " " << intersection[index] << std::endl;
  }

  // book the vectors of values
  alignmentTrend dxyPhiMeans_;
  alignmentTrend dxyPhiChi2_;
  alignmentTrend dxyPhiHiErr_;
  alignmentTrend dxyPhiLoErr_;
  alignmentTrend dxyPhiKS_;
  alignmentTrend dxyPhiHi_;
  alignmentTrend dxyPhiLo_;

  alignmentTrend dxyEtaMeans_;
  alignmentTrend dxyEtaChi2_;
  alignmentTrend dxyEtaHiErr_;
  alignmentTrend dxyEtaLoErr_;
  alignmentTrend dxyEtaKS_;
  alignmentTrend dxyEtaHi_;
  alignmentTrend dxyEtaLo_;

  alignmentTrend dzPhiMeans_;
  alignmentTrend dzPhiChi2_;
  alignmentTrend dzPhiHiErr_;
  alignmentTrend dzPhiLoErr_;
  alignmentTrend dzPhiKS_;
  alignmentTrend dzPhiHi_;
  alignmentTrend dzPhiLo_;

  alignmentTrend dzEtaMeans_;
  alignmentTrend dzEtaChi2_;
  alignmentTrend dzEtaHiErr_;
  alignmentTrend dzEtaLoErr_;
  alignmentTrend dzEtaKS_;
  alignmentTrend dzEtaHi_;
  alignmentTrend dzEtaLo_;

  // unrolled histos

  std::map<TString, std::vector<unrolledHisto> > dxyVect;
  std::map<TString, std::vector<unrolledHisto> > dzVect;

  double lumiSoFar = 0.0;

  // loop over the runs in the intersection
  //unsigned int last = (DEBUG==true) ? 50 : intersection.size();

  //std::function<void()> f_processData = std::bind(processData,1,intersection,nDirs_,dirs,LegLabels,lumiSoFar,runs,lumiByRun,lumiMapByRun,useRMS,dxyPhiMeans_,dxyPhiHi_,dxyPhiLo_,dxyEtaMeans_,dxyEtaHi_,dxyEtaLo_,dzPhiMeans_,dzPhiHi_,dzPhiLo_,dzEtaMeans_,dzEtaHi_,dzEtaLo_,dxyVect,dzVect);

  std::cout << " pre do-stuff: " << runs.size() << std::endl;

  //we should use std::bind to create a functor and then pass it to the procPool
  auto f_processData = std::bind(processData, _1, intersection, nDirs_, dirs, LegLabels, useRMS);

  //f_processData(0);
  //std::cout<<" post do-stuff: " <<  runs.size() << std::endl;

  TProcPool procPool(std::min(nWorkers, intersection.size()));
  std::vector<size_t> range(std::min(nWorkers, intersection.size()));
  std::iota(range.begin(), range.end(), 0);
  //procPool.Map([&f_processData](size_t a) { f_processData(a); },{1,2,3});
  auto extracts = procPool.Map(f_processData, range);

  // sort the extracts according to the global index
  std::sort(extracts.begin(), extracts.end(), [](const outTrends &a, const outTrends &b) -> bool {
    return a.m_index < b.m_index;
  });

  // re-assemble everything together
  for (auto extractedTrend : extracts) {
    std::cout << "lumiSoFar: " << lumiSoFar << "/fb" << std::endl;

    runs.insert(std::end(runs), std::begin(extractedTrend.m_runs), std::end(extractedTrend.m_runs));

    // luminosity needs a different treatment
    // we need to re-sum the luminosity so far

    for (const auto &run : extractedTrend.m_runs) {
      std::cout << run << " " << lumiSoFar + extractedTrend.m_lumiMapByRun[run] << std::endl;
      lumiByRun.push_back(lumiSoFar + extractedTrend.m_lumiMapByRun[run]);
      lumiMapByRun[run] = (lumiSoFar + extractedTrend.m_lumiMapByRun[run]);
    }

    lumiSoFar += (extractedTrend.m_lumiSoFar / 1000.);

    /*
      lumiByRun.insert(std::end(lumiByRun), std::begin(extractedTrend.m_lumiByRun), std::end(extractedTrend.m_lumiByRun));
      lumiMapByRun.insert(extractedTrend.m_lumiMapByRun.begin(),extractedTrend.m_lumiMapByRun.end());
    */

    for (const auto &label : LegLabels) {
      //******************************//
      dxyPhiMeans_[label].insert(std::end(dxyPhiMeans_[label]),
                                 std::begin(extractedTrend.m_dxyPhiMeans[label]),
                                 std::end(extractedTrend.m_dxyPhiMeans[label]));
      dxyPhiChi2_[label].insert(std::end(dxyPhiChi2_[label]),
                                std::begin(extractedTrend.m_dxyPhiChi2[label]),
                                std::end(extractedTrend.m_dxyPhiChi2[label]));
      dxyPhiKS_[label].insert(std::end(dxyPhiKS_[label]),
                              std::begin(extractedTrend.m_dxyPhiKS[label]),
                              std::end(extractedTrend.m_dxyPhiKS[label]));

      dxyPhiHi_[label].insert(std::end(dxyPhiHi_[label]),
                              std::begin(extractedTrend.m_dxyPhiHi[label]),
                              std::end(extractedTrend.m_dxyPhiHi[label]));
      dxyPhiLo_[label].insert(std::end(dxyPhiLo_[label]),
                              std::begin(extractedTrend.m_dxyPhiLo[label]),
                              std::end(extractedTrend.m_dxyPhiLo[label]));

      //******************************//
      dzPhiMeans_[label].insert(std::end(dzPhiMeans_[label]),
                                std::begin(extractedTrend.m_dzPhiMeans[label]),
                                std::end(extractedTrend.m_dzPhiMeans[label]));
      dzPhiChi2_[label].insert(std::end(dzPhiChi2_[label]),
                               std::begin(extractedTrend.m_dzPhiChi2[label]),
                               std::end(extractedTrend.m_dzPhiChi2[label]));
      dzPhiKS_[label].insert(std::end(dzPhiKS_[label]),
                             std::begin(extractedTrend.m_dzPhiKS[label]),
                             std::end(extractedTrend.m_dzPhiKS[label]));

      dzPhiHi_[label].insert(std::end(dzPhiHi_[label]),
                             std::begin(extractedTrend.m_dzPhiHi[label]),
                             std::end(extractedTrend.m_dzPhiHi[label]));
      dzPhiLo_[label].insert(std::end(dzPhiLo_[label]),
                             std::begin(extractedTrend.m_dzPhiLo[label]),
                             std::end(extractedTrend.m_dzPhiLo[label]));

      //******************************//
      dxyEtaMeans_[label].insert(std::end(dxyEtaMeans_[label]),
                                 std::begin(extractedTrend.m_dxyEtaMeans[label]),
                                 std::end(extractedTrend.m_dxyEtaMeans[label]));
      dxyEtaChi2_[label].insert(std::end(dxyEtaChi2_[label]),
                                std::begin(extractedTrend.m_dxyEtaChi2[label]),
                                std::end(extractedTrend.m_dxyEtaChi2[label]));
      dxyEtaKS_[label].insert(std::end(dxyEtaKS_[label]),
                              std::begin(extractedTrend.m_dxyEtaKS[label]),
                              std::end(extractedTrend.m_dxyEtaKS[label]));

      dxyEtaHi_[label].insert(std::end(dxyEtaHi_[label]),
                              std::begin(extractedTrend.m_dxyEtaHi[label]),
                              std::end(extractedTrend.m_dxyEtaHi[label]));
      dxyEtaLo_[label].insert(std::end(dxyEtaLo_[label]),
                              std::begin(extractedTrend.m_dxyEtaLo[label]),
                              std::end(extractedTrend.m_dxyEtaLo[label]));

      //******************************//
      dzEtaMeans_[label].insert(std::end(dzEtaMeans_[label]),
                                std::begin(extractedTrend.m_dzEtaMeans[label]),
                                std::end(extractedTrend.m_dzEtaMeans[label]));
      dzEtaChi2_[label].insert(std::end(dzEtaChi2_[label]),
                               std::begin(extractedTrend.m_dzEtaChi2[label]),
                               std::end(extractedTrend.m_dzEtaChi2[label]));
      dzEtaKS_[label].insert(std::end(dzEtaKS_[label]),
                             std::begin(extractedTrend.m_dzEtaKS[label]),
                             std::end(extractedTrend.m_dzEtaKS[label]));

      dzEtaHi_[label].insert(std::end(dzEtaHi_[label]),
                             std::begin(extractedTrend.m_dzEtaHi[label]),
                             std::end(extractedTrend.m_dzEtaHi[label]));
      dzEtaLo_[label].insert(std::end(dzEtaLo_[label]),
                             std::begin(extractedTrend.m_dzEtaLo[label]),
                             std::end(extractedTrend.m_dzEtaLo[label]));

      //******************************//
      dxyVect[label].insert(std::end(dxyVect[label]),
                            std::begin(extractedTrend.m_dxyVect[label]),
                            std::end(extractedTrend.m_dxyVect[label]));
      dzVect[label].insert(std::end(dzVect[label]),
                           std::begin(extractedTrend.m_dzVect[label]),
                           std::end(extractedTrend.m_dzVect[label]));
    }
  }

  // extra vectors for low and high boundaries

  for (const auto &label : LegLabels) {
    for (unsigned int it = 0; it < dxyPhiMeans_[label].size(); it++) {
      dxyPhiHiErr_[label].push_back(std::abs(dxyPhiHi_[label][it] - dxyPhiMeans_[label][it]));
      dxyPhiLoErr_[label].push_back(std::abs(dxyPhiLo_[label][it] - dxyPhiMeans_[label][it]));
      dxyEtaHiErr_[label].push_back(std::abs(dxyEtaHi_[label][it] - dxyEtaMeans_[label][it]));
      dxyEtaLoErr_[label].push_back(std::abs(dxyEtaLo_[label][it] - dxyEtaMeans_[label][it]));

      std::cout << "label: " << label << " means:" << dxyEtaMeans_[label][it] << " low: " << dxyEtaLo_[label][it]
                << " loErr: " << dxyEtaLoErr_[label][it] << std::endl;

      dzPhiHiErr_[label].push_back(std::abs(dzPhiHi_[label][it] - dzPhiMeans_[label][it]));
      dzPhiLoErr_[label].push_back(std::abs(dzPhiLo_[label][it] - dzPhiMeans_[label][it]));
      dzEtaHiErr_[label].push_back(std::abs(dzEtaHi_[label][it] - dzEtaMeans_[label][it]));
      dzEtaLoErr_[label].push_back(std::abs(dzEtaLo_[label][it] - dzEtaMeans_[label][it]));
    }
  }

  // just check runs are ordered
  //for(const auto &run : runs){
  //  std::cout<<" "<< run ;
  // }

  // main function call
  /*
    processData(0,intersection,nDirs_,dirs,LegLabels,lumiSoFar,runs,lumiByRun,lumiMapByRun,useRMS,
    dxyPhiMeans_,dxyPhiHi_,dxyPhiLo_,	 	                  
    dxyEtaMeans_,dxyEtaHi_,dxyEtaLo_,	 	                  
    dzPhiMeans_,dzPhiHi_,dzPhiLo_,	 	  
    dzEtaMeans_,dzEtaHi_,dzEtaLo_,       
    dxyVect,dzVect
    );
  */

  // do the trend-plotting!

  TCanvas *c_dxy_phi_vs_run = new TCanvas("c_dxy_phi_vs_run", "dxy(#phi) bias vs run number", 2000, 800);
  TCanvas *c_dxy_eta_vs_run = new TCanvas("c_dxy_eta_vs_run", "dxy(#eta) bias vs run number", 2000, 800);
  TCanvas *c_dz_phi_vs_run = new TCanvas("c_dz_phi_vs_run", "dz(#phi) bias vs run number", 2000, 800);
  TCanvas *c_dz_eta_vs_run = new TCanvas("c_dz_eta_vs_run", "dz(#eta) bias vs run number", 2000, 800);

  TCanvas *c_RMS_dxy_phi_vs_run = new TCanvas("c_RMS_dxy_phi_vs_run", "dxy(#phi) bias vs run number", 2000, 800);
  TCanvas *c_RMS_dxy_eta_vs_run = new TCanvas("c_RMS_dxy_eta_vs_run", "dxy(#eta) bias vs run number", 2000, 800);
  TCanvas *c_RMS_dz_phi_vs_run = new TCanvas("c_RMS_dz_phi_vs_run", "dxy(#phi) bias vs run number", 2000, 800);
  TCanvas *c_RMS_dz_eta_vs_run = new TCanvas("c_RMS_dz_eta_vs_run", "dxy(#eta) bias vs run number", 2000, 800);

  TCanvas *c_mean_dxy_phi_vs_run = new TCanvas("c_mean_dxy_phi_vs_run", "dxy(#phi) bias vs run number", 2000, 800);
  TCanvas *c_mean_dxy_eta_vs_run = new TCanvas("c_mean_dxy_eta_vs_run", "dxy(#eta) bias vs run number", 2000, 800);
  TCanvas *c_mean_dz_phi_vs_run = new TCanvas("c_mean_dz_phi_vs_run", "dxy(#phi) bias vs run number", 2000, 800);
  TCanvas *c_mean_dz_eta_vs_run = new TCanvas("c_mean_dz_eta_vs_run", "dxy(#eta) bias vs run number", 2000, 800);

  TCanvas *Scatter_dxy_vs_run = new TCanvas("Scatter_dxy_vs_run", "dxy bias vs run number", 2000, 800);
  Scatter_dxy_vs_run->Divide(1, nDirs_);
  TCanvas *Scatter_dz_vs_run = new TCanvas("Scatter_dz_vs_run", "dxy bias vs run number", 2000, 800);
  Scatter_dz_vs_run->Divide(1, nDirs_);

  TCanvas *c_chisquare_vs_run = new TCanvas("c_chisquare_vs_run", "chi2 of pol0 fit vs run number", 2000, 800);
  c_chisquare_vs_run->Divide(2, 2);

  TCanvas *c_KSScore_vs_run = new TCanvas("c_KSScore_vs_run", "KS score compatibility to 0 vs run number", 2000, 1000);
  c_KSScore_vs_run->Divide(2, 2);

  // bias on the mean

  TGraph *g_dxy_phi_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dxy_phi_vs_run[nDirs_];
  TGraph *g_chi2_dxy_phi_vs_run[nDirs_];
  TGraph *g_KS_dxy_phi_vs_run[nDirs_];
  TGraph *gprime_dxy_phi_vs_run[nDirs_];
  TGraph *g_dxy_phi_hi_vs_run[nDirs_];
  TGraph *g_dxy_phi_lo_vs_run[nDirs_];

  TGraph *g_dxy_eta_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dxy_eta_vs_run[nDirs_];
  TGraph *g_chi2_dxy_eta_vs_run[nDirs_];
  TGraph *g_KS_dxy_eta_vs_run[nDirs_];
  TGraph *gprime_dxy_eta_vs_run[nDirs_];
  TGraph *g_dxy_eta_hi_vs_run[nDirs_];
  TGraph *g_dxy_eta_lo_vs_run[nDirs_];

  TGraph *g_dz_phi_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dz_phi_vs_run[nDirs_];
  TGraph *g_chi2_dz_phi_vs_run[nDirs_];
  TGraph *g_KS_dz_phi_vs_run[nDirs_];
  TGraph *gprime_dz_phi_vs_run[nDirs_];
  TGraph *g_dz_phi_hi_vs_run[nDirs_];
  TGraph *g_dz_phi_lo_vs_run[nDirs_];

  TGraph *g_dz_eta_vs_run[nDirs_];
  TGraphAsymmErrors *gerr_dz_eta_vs_run[nDirs_];
  TGraph *g_chi2_dz_eta_vs_run[nDirs_];
  TGraph *g_KS_dz_eta_vs_run[nDirs_];
  TGraph *gprime_dz_eta_vs_run[nDirs_];
  TGraph *g_dz_eta_hi_vs_run[nDirs_];
  TGraph *g_dz_eta_lo_vs_run[nDirs_];

  // resolutions

  TH1F *h_RMS_dxy_phi_vs_run[nDirs_];
  TH1F *h_RMS_dxy_eta_vs_run[nDirs_];
  TH1F *h_RMS_dz_phi_vs_run[nDirs_];
  TH1F *h_RMS_dz_eta_vs_run[nDirs_];

  // scatters of integrated bias

  TH2F *h2_scatter_dxy_vs_run[nDirs_];
  TH2F *h2_scatter_dz_vs_run[nDirs_];

  // decide the type

  TString theType = "";
  TString theTypeLabel = "";
  if (lumi_axis_format) {
    theType = "luminosity";
    theTypeLabel = "Processed luminosity [1/fb]";
    x_ticks = lumiByRun;
  } else {
    if (!time_axis_format) {
      theType = "run number";
      theTypeLabel = "run number";
      x_ticks = runs;
    } else {
      theType = "date";
      theTypeLabel = "UTC date";
      for (const auto &run : runs) {
        x_ticks.push_back(times[run].Convert());
      }
    }
  }

  //TLegend *my_lego = new TLegend(0.08,0.80,0.18,0.93);
  //my_lego-> SetNColumns(2);

  TLegend *my_lego = new TLegend(0.75, 0.76, 0.92, 0.93);
  //TLegend *my_lego = new TLegend(0.75,0.26,0.92,0.43);
  //my_lego->SetHeader("2017 Data","C");
  TLine *line = new TLine(0, 0, 1, 0);
  line->SetLineColor(kBlue);
  line->SetLineStyle(9);
  line->SetLineWidth(1);
  my_lego->AddEntry(line, "pixel calibration update", "L");
  my_lego->SetFillColor(10);
  my_lego->SetTextSize(0.042);
  my_lego->SetTextFont(42);
  my_lego->SetFillColor(10);
  my_lego->SetLineColor(10);
  my_lego->SetShadowColor(10);

  // arrays for storing RMS histograms
  TObjArray *arr_dxy_phi = new TObjArray();
  TObjArray *arr_dz_phi = new TObjArray();
  TObjArray *arr_dxy_eta = new TObjArray();
  TObjArray *arr_dz_eta = new TObjArray();

  arr_dxy_phi->Expand(nDirs_);
  arr_dz_phi->Expand(nDirs_);
  arr_dxy_eta->Expand(nDirs_);
  arr_dz_eta->Expand(nDirs_);

  int ccc = 0;
  for (const auto &tick : x_ticks) {
    outfile << "***************************************" << std::endl;
    outfile << tick << std::endl;
    for (Int_t j = 0; j < nDirs_; j++) {
      double RMSxyphi = std::abs(dxyPhiLo_[LegLabels[j]][ccc] - dxyPhiHi_[LegLabels[j]][ccc]);
      double RMSxyeta = std::abs(dxyEtaLo_[LegLabels[j]][ccc] - dxyEtaHi_[LegLabels[j]][ccc]);
      double RMSzphi = std::abs(dzPhiLo_[LegLabels[j]][ccc] - dzPhiHi_[LegLabels[j]][ccc]);
      double RMSzeta = std::abs(dzEtaLo_[LegLabels[j]][ccc] - dzEtaHi_[LegLabels[j]][ccc]);

      if (RMSxyphi > 100 || RMSxyeta > 100 || RMSzphi > 100 || RMSzeta > 100) {
        outfile << LegLabels[j] << " dxy(phi) " << dxyPhiMeans_[LegLabels[j]][ccc] << " "
                << dxyPhiLo_[LegLabels[j]][ccc] << " " << dxyPhiHi_[LegLabels[j]][ccc] << " "
                << std::abs(dxyPhiLo_[LegLabels[j]][ccc] - dxyPhiHi_[LegLabels[j]][ccc]) << std::endl;
        outfile << LegLabels[j] << " dxy(eta) " << dxyEtaMeans_[LegLabels[j]][ccc] << " "
                << dxyEtaLo_[LegLabels[j]][ccc] << " " << dxyEtaHi_[LegLabels[j]][ccc] << " "
                << std::abs(dxyEtaLo_[LegLabels[j]][ccc] - dxyEtaHi_[LegLabels[j]][ccc]) << std::endl;
        outfile << LegLabels[j] << " dz (phi) " << dzPhiMeans_[LegLabels[j]][ccc] << " " << dzPhiLo_[LegLabels[j]][ccc]
                << " " << dzPhiHi_[LegLabels[j]][ccc] << " "
                << std::abs(dzPhiLo_[LegLabels[j]][ccc] - dzPhiHi_[LegLabels[j]][ccc]) << std::endl;
        outfile << LegLabels[j] << " dz (eta) " << dzEtaMeans_[LegLabels[j]][ccc] << " " << dzEtaLo_[LegLabels[j]][ccc]
                << " " << dzEtaHi_[LegLabels[j]][ccc] << " "
                << std::abs(dzEtaLo_[LegLabels[j]][ccc] - dzEtaHi_[LegLabels[j]][ccc]) << std::endl;
      }
    }
    ccc++;
  }
  outfile << std::endl;

  pv::bundle theBundle =
      pv::bundle(nDirs_, theType, theTypeLabel, lumiMapByRun, times, lumi_axis_format, time_axis_format, useRMS);
  theBundle.printAll();

  for (Int_t j = 0; j < nDirs_; j++) {
    // check on the sanity
    std::cout << "x_ticks.size()= " << x_ticks.size() << " dxyPhiMeans_[LegLabels[" << j
              << "]].size()=" << dxyPhiMeans_[LegLabels[j]].size() << std::endl;

    // *************************************
    // dxy vs phi
    // *************************************

    auto dxyPhiInputs =
        pv::wrappedTrends(dxyPhiMeans_, dxyPhiLo_, dxyPhiHi_, dxyPhiLoErr_, dxyPhiHiErr_, dxyPhiChi2_, dxyPhiKS_);

    outputGraphs(dxyPhiInputs,
                 x_ticks,
                 ex_ticks,
                 c_dxy_phi_vs_run,
                 c_mean_dxy_phi_vs_run,
                 c_RMS_dxy_phi_vs_run,
                 g_dxy_phi_vs_run[j],
                 g_chi2_dxy_phi_vs_run[j],
                 g_KS_dxy_phi_vs_run[j],
                 g_dxy_phi_lo_vs_run[j],
                 g_dxy_phi_hi_vs_run[j],
                 gerr_dxy_phi_vs_run[j],
                 h_RMS_dxy_phi_vs_run,
                 theBundle,
                 pv::dxyphi,
                 j,
                 arr_dxy_phi,
                 LegLabels[j],
                 my_lego);

    // *************************************
    // dxy vs eta
    // *************************************

    auto dxyEtaInputs =
        pv::wrappedTrends(dxyEtaMeans_, dxyEtaLo_, dxyEtaHi_, dxyEtaLoErr_, dxyEtaHiErr_, dxyEtaChi2_, dxyEtaKS_);

    outputGraphs(dxyEtaInputs,
                 x_ticks,
                 ex_ticks,
                 c_dxy_eta_vs_run,
                 c_mean_dxy_eta_vs_run,
                 c_RMS_dxy_eta_vs_run,
                 g_dxy_eta_vs_run[j],
                 g_chi2_dxy_eta_vs_run[j],
                 g_KS_dxy_eta_vs_run[j],
                 g_dxy_eta_lo_vs_run[j],
                 g_dxy_eta_hi_vs_run[j],
                 gerr_dxy_eta_vs_run[j],
                 h_RMS_dxy_eta_vs_run,
                 theBundle,
                 pv::dxyeta,
                 j,
                 arr_dxy_eta,
                 LegLabels[j],
                 my_lego);

    // *************************************
    // dz vs phi
    // *************************************

    auto dzPhiInputs =
        pv::wrappedTrends(dzPhiMeans_, dzPhiLo_, dzPhiHi_, dzPhiLoErr_, dzPhiHiErr_, dzPhiChi2_, dzPhiKS_);

    outputGraphs(dzPhiInputs,
                 x_ticks,
                 ex_ticks,
                 c_dz_phi_vs_run,
                 c_mean_dz_phi_vs_run,
                 c_RMS_dz_phi_vs_run,
                 g_dz_phi_vs_run[j],
                 g_chi2_dz_phi_vs_run[j],
                 g_KS_dz_phi_vs_run[j],
                 g_dz_phi_lo_vs_run[j],
                 g_dz_phi_hi_vs_run[j],
                 gerr_dz_phi_vs_run[j],
                 h_RMS_dz_phi_vs_run,
                 theBundle,
                 pv::dzphi,
                 j,
                 arr_dz_phi,
                 LegLabels[j],
                 my_lego);

    // *************************************
    // dz vs eta
    // *************************************

    auto dzEtaInputs =
        pv::wrappedTrends(dzEtaMeans_, dzEtaLo_, dzEtaHi_, dzEtaLoErr_, dzEtaHiErr_, dzEtaChi2_, dzEtaKS_);

    outputGraphs(dzEtaInputs,
                 x_ticks,
                 ex_ticks,
                 c_dz_eta_vs_run,
                 c_mean_dz_eta_vs_run,
                 c_RMS_dz_eta_vs_run,
                 g_dz_eta_vs_run[j],
                 g_chi2_dz_eta_vs_run[j],
                 g_KS_dz_eta_vs_run[j],
                 g_dz_eta_lo_vs_run[j],
                 g_dz_eta_hi_vs_run[j],
                 gerr_dz_eta_vs_run[j],
                 h_RMS_dz_eta_vs_run,
                 theBundle,
                 pv::dzeta,
                 j,
                 arr_dz_eta,
                 LegLabels[j],
                 my_lego);

    // *************************************
    // Integrated bias dxy scatter plots
    // *************************************

    h2_scatter_dxy_vs_run[j] =
        new TH2F(Form("h2_scatter_dxy_%s", LegLabels[j].Data()),
                 Form("scatter of d_{xy} vs %s;%s;d_{xy} [cm]", theType.Data(), theTypeLabel.Data()),
                 x_ticks.size() - 1,
                 &(x_ticks[0]),
                 dxyVect[LegLabels[j]][0].get_n_bins(),
                 dxyVect[LegLabels[j]][0].get_y_min(),
                 dxyVect[LegLabels[j]][0].get_y_max());
    h2_scatter_dxy_vs_run[j]->SetStats(kFALSE);

    for (unsigned int runindex = 0; runindex < x_ticks.size(); runindex++) {
      for (unsigned int binindex = 0; binindex < dxyVect[LegLabels[j]][runindex].get_n_bins(); binindex++) {
        h2_scatter_dxy_vs_run[j]->SetBinContent(runindex + 1,
                                                binindex + 1,
                                                dxyVect[LegLabels[j]][runindex].get_bin_contents().at(binindex) /
                                                    dxyVect[LegLabels[j]][runindex].get_integral());
      }
    }

    //Scatter_dxy_vs_run->cd();
    h2_scatter_dxy_vs_run[j]->SetFillColorAlpha(pv::colors[j], 0.3);
    h2_scatter_dxy_vs_run[j]->SetMarkerColor(pv::colors[j]);
    h2_scatter_dxy_vs_run[j]->SetLineColor(pv::colors[j]);
    h2_scatter_dxy_vs_run[j]->SetMarkerStyle(pv::markers[j]);

    auto h_dxypfx_tmp = (TProfile *)(((TH2F *)h2_scatter_dxy_vs_run[j])->ProfileX(Form("_apfx_%i", j), 1, -1, "o"));
    h_dxypfx_tmp->SetName(TString(h2_scatter_dxy_vs_run[j]->GetName()) + "_pfx");
    h_dxypfx_tmp->SetStats(kFALSE);
    h_dxypfx_tmp->SetMarkerColor(pv::colors[j]);
    h_dxypfx_tmp->SetLineColor(pv::colors[j]);
    h_dxypfx_tmp->SetLineWidth(2);
    h_dxypfx_tmp->SetMarkerSize(1);
    h_dxypfx_tmp->SetMarkerStyle(pv::markers[j]);

    beautify(h2_scatter_dxy_vs_run[j]);
    beautify(h_dxypfx_tmp);

    Scatter_dxy_vs_run->cd(j + 1);
    adjustmargins(Scatter_dxy_vs_run->cd(j + 1));
    //h_dxypfx_tmp->GetYaxis()->SetRangeUser(-0.01,0.01);
    //h2_scatter_dxy_vs_run[j]->GetYaxis()->SetRangeUser(-0.5,0.5);
    h2_scatter_dxy_vs_run[j]->Draw("colz");
    h_dxypfx_tmp->Draw("same");

    // *************************************
    // Integrated bias dz scatter plots
    // *************************************

    h2_scatter_dz_vs_run[j] =
        new TH2F(Form("h2_scatter_dz_%s", LegLabels[j].Data()),
                 Form("scatter of d_{z} vs %s;%s;d_{z} [cm]", theType.Data(), theTypeLabel.Data()),
                 x_ticks.size() - 1,
                 &(x_ticks[0]),
                 dzVect[LegLabels[j]][0].get_n_bins(),
                 dzVect[LegLabels[j]][0].get_y_min(),
                 dzVect[LegLabels[j]][0].get_y_max());
    h2_scatter_dz_vs_run[j]->SetStats(kFALSE);

    for (unsigned int runindex = 0; runindex < x_ticks.size(); runindex++) {
      for (unsigned int binindex = 0; binindex < dzVect[LegLabels[j]][runindex].get_n_bins(); binindex++) {
        h2_scatter_dz_vs_run[j]->SetBinContent(runindex + 1,
                                               binindex + 1,
                                               dzVect[LegLabels[j]][runindex].get_bin_contents().at(binindex) /
                                                   dzVect[LegLabels[j]][runindex].get_integral());
      }
    }

    //Scatter_dz_vs_run->cd();
    h2_scatter_dz_vs_run[j]->SetFillColorAlpha(pv::colors[j], 0.3);
    h2_scatter_dz_vs_run[j]->SetMarkerColor(pv::colors[j]);
    h2_scatter_dz_vs_run[j]->SetLineColor(pv::colors[j]);
    h2_scatter_dz_vs_run[j]->SetMarkerStyle(pv::markers[j]);

    auto h_dzpfx_tmp = (TProfile *)(((TH2F *)h2_scatter_dz_vs_run[j])->ProfileX(Form("_apfx_%i", j), 1, -1, "o"));
    h_dzpfx_tmp->SetName(TString(h2_scatter_dz_vs_run[j]->GetName()) + "_pfx");
    h_dzpfx_tmp->SetStats(kFALSE);
    h_dzpfx_tmp->SetMarkerColor(pv::colors[j]);
    h_dzpfx_tmp->SetLineColor(pv::colors[j]);
    h_dzpfx_tmp->SetLineWidth(2);
    h_dzpfx_tmp->SetMarkerSize(1);
    h_dzpfx_tmp->SetMarkerStyle(pv::markers[j]);

    beautify(h2_scatter_dz_vs_run[j]);
    beautify(h_dzpfx_tmp);

    Scatter_dz_vs_run->cd(j + 1);
    adjustmargins(Scatter_dz_vs_run->cd(j + 1));
    //h_dzpfx_tmp->GetYaxis()->SetRangeUser(-0.01,0.01);
    //h2_scatter_dz_vs_run[j]->GetYaxis()->SetRangeUser(-0.5,0.5);
    h2_scatter_dz_vs_run[j]->Draw("colz");
    h_dzpfx_tmp->Draw("same");

    // ****************************************
    // Canvas for chi2 goodness of pol0 fit
    // ****************************************

    // 1st pad
    c_chisquare_vs_run->cd(1);
    adjustmargins(c_chisquare_vs_run->cd(1));
    g_chi2_dxy_phi_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_chi2_dxy_phi_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_chi2_dxy_phi_vs_run[j]->SetLineColor(pv::colors[j]);

    g_chi2_dxy_phi_vs_run[j]->SetName(Form("g_chi2_dxy_phi_%s", LegLabels[j].Data()));
    g_chi2_dxy_phi_vs_run[j]->SetTitle(Form("log_{10}(#chi2/ndf) of d_{xy}(#varphi) fit vs %s", theType.Data()));
    g_chi2_dxy_phi_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_chi2_dxy_phi_vs_run[j]->GetYaxis()->SetTitle("log_{10}(#chi^{2}/ndf) of d_{xy}(#phi) pol0 fit");
    g_chi2_dxy_phi_vs_run[j]->GetYaxis()->SetRangeUser(-0.5, 4.5);
    if (lumi_axis_format) {
      g_chi2_dxy_phi_vs_run[j]->GetXaxis()->SetRangeUser(0., theBundle.getTotalLumi());
    }
    beautify(g_chi2_dxy_phi_vs_run[j]);
    //g_chi2_dxy_phi_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_chi2_dxy_phi_vs_run[j]->Draw("APL");
    } else {
      g_chi2_dxy_phi_vs_run[j]->Draw("PLsame");
    }

    if (time_axis_format) {
      timify(g_chi2_dxy_phi_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    auto current_pad = static_cast<TCanvas *>(gPad);
    superImposeIOVBoundaries(current_pad, lumi_axis_format, time_axis_format, lumiMapByRun, times, false);

    // 2nd pad
    c_chisquare_vs_run->cd(2);
    adjustmargins(c_chisquare_vs_run->cd(2));
    g_chi2_dxy_eta_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_chi2_dxy_eta_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_chi2_dxy_eta_vs_run[j]->SetLineColor(pv::colors[j]);

    g_chi2_dxy_eta_vs_run[j]->SetName(Form("g_chi2_dxy_eta_%s", LegLabels[j].Data()));
    g_chi2_dxy_eta_vs_run[j]->SetTitle(Form("log_{10}(#chi2/ndf) of d_{xy}(#eta) fit vs %s", theType.Data()));
    g_chi2_dxy_eta_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_chi2_dxy_eta_vs_run[j]->GetYaxis()->SetTitle("log_{10}(#chi^{2}/ndf) of d_{xy}(#eta) pol0 fit");
    g_chi2_dxy_eta_vs_run[j]->GetYaxis()->SetRangeUser(-0.5, 4.5);
    if (lumi_axis_format) {
      g_chi2_dxy_eta_vs_run[j]->GetXaxis()->SetRangeUser(0., theBundle.getTotalLumi());
    }
    beautify(g_chi2_dxy_eta_vs_run[j]);
    //g_chi2_dxy_eta_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_chi2_dxy_eta_vs_run[j]->Draw("APL");
    } else {
      g_chi2_dxy_eta_vs_run[j]->Draw("PLsame");
    }

    if (time_axis_format) {
      timify(g_chi2_dxy_eta_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    current_pad = static_cast<TCanvas *>(gPad);
    superImposeIOVBoundaries(current_pad, lumi_axis_format, time_axis_format, lumiMapByRun, times, false);

    //3d pad
    c_chisquare_vs_run->cd(3);
    adjustmargins(c_chisquare_vs_run->cd(3));
    g_chi2_dz_phi_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_chi2_dz_phi_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_chi2_dz_phi_vs_run[j]->SetLineColor(pv::colors[j]);

    g_chi2_dz_phi_vs_run[j]->SetName(Form("g_chi2_dz_phi_%s", LegLabels[j].Data()));
    g_chi2_dz_phi_vs_run[j]->SetTitle(Form("log_{10}(#chi2/ndf) of d_{z}(#varphi) fit vs %s", theType.Data()));
    g_chi2_dz_phi_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_chi2_dz_phi_vs_run[j]->GetYaxis()->SetTitle("log_{10}(#chi^{2}/ndf) of d_{z}(#phi) pol0 fit");
    g_chi2_dz_phi_vs_run[j]->GetYaxis()->SetRangeUser(-0.5, 4.5);
    if (lumi_axis_format) {
      g_chi2_dz_phi_vs_run[j]->GetXaxis()->SetRangeUser(0., theBundle.getTotalLumi());
    }
    beautify(g_chi2_dz_phi_vs_run[j]);
    //g_chi2_dz_phi_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_chi2_dz_phi_vs_run[j]->Draw("APL");
    } else {
      g_chi2_dz_phi_vs_run[j]->Draw("PLsame");
    }

    if (time_axis_format) {
      timify(g_chi2_dz_phi_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    current_pad = static_cast<TCanvas *>(gPad);
    superImposeIOVBoundaries(current_pad, lumi_axis_format, time_axis_format, lumiMapByRun, times, false);

    //4th pad
    c_chisquare_vs_run->cd(4);
    adjustmargins(c_chisquare_vs_run->cd(4));
    g_chi2_dz_eta_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_chi2_dz_eta_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_chi2_dz_eta_vs_run[j]->SetLineColor(pv::colors[j]);

    g_chi2_dz_eta_vs_run[j]->SetName(Form("g_chi2_dz_eta_%s", LegLabels[j].Data()));
    g_chi2_dz_eta_vs_run[j]->SetTitle(Form("log_{10}(#chi2/ndf) of d_{z}(#eta) fit vs %s", theType.Data()));
    g_chi2_dz_eta_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_chi2_dz_eta_vs_run[j]->GetYaxis()->SetTitle("log_{10}(#chi^{2}/ndf) of d_{z}(#eta) pol0 fit");
    g_chi2_dz_eta_vs_run[j]->GetYaxis()->SetRangeUser(-0.5, 4.5);
    if (lumi_axis_format) {
      g_chi2_dz_eta_vs_run[j]->GetXaxis()->SetRangeUser(0., theBundle.getTotalLumi());
    }
    beautify(g_chi2_dz_eta_vs_run[j]);
    //g_chi2_dz_eta_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_chi2_dz_eta_vs_run[j]->Draw("APL");
    } else {
      g_chi2_dz_eta_vs_run[j]->Draw("PLsame");
    }

    if (time_axis_format) {
      timify(g_chi2_dz_eta_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    current_pad = static_cast<TCanvas *>(gPad);
    superImposeIOVBoundaries(current_pad, lumi_axis_format, time_axis_format, lumiMapByRun, times, false);

    // ****************************************
    // Canvas for Kolmogorov-Smirnov test
    // ****************************************

    // 1st pad
    c_KSScore_vs_run->cd(1);
    adjustmargins(c_KSScore_vs_run->cd(1));
    g_KS_dxy_phi_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_KS_dxy_phi_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_KS_dxy_phi_vs_run[j]->SetLineColor(pv::colors[j]);

    g_KS_dxy_phi_vs_run[j]->SetName(Form("g_KS_dxy_phi_%s", LegLabels[j].Data()));
    g_KS_dxy_phi_vs_run[j]->SetTitle(Form("log_{10}(KS-score) of d_{xy}(#varphi) vs %s", theType.Data()));
    g_KS_dxy_phi_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_KS_dxy_phi_vs_run[j]->GetYaxis()->SetTitle("log_{10}(KS-score) of d_{xy}(#phi) w.r.t 0");
    g_KS_dxy_phi_vs_run[j]->GetYaxis()->SetRangeUser(-20., 1.);
    beautify(g_KS_dxy_phi_vs_run[j]);
    //g_KS_dxy_phi_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_KS_dxy_phi_vs_run[j]->Draw("AP");
    } else {
      g_KS_dxy_phi_vs_run[j]->Draw("Psame");
    }

    if (time_axis_format) {
      timify(g_KS_dxy_phi_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    // 2nd pad
    c_KSScore_vs_run->cd(2);
    adjustmargins(c_KSScore_vs_run->cd(2));
    g_KS_dxy_eta_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_KS_dxy_eta_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_KS_dxy_eta_vs_run[j]->SetLineColor(pv::colors[j]);

    g_KS_dxy_eta_vs_run[j]->SetName(Form("g_KS_dxy_eta_%s", LegLabels[j].Data()));
    g_KS_dxy_eta_vs_run[j]->SetTitle(Form("log_{10}(KS-score) of d_{xy}(#eta) vs %s", theType.Data()));
    g_KS_dxy_eta_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_KS_dxy_eta_vs_run[j]->GetYaxis()->SetTitle("log_{10}(KS-score) of d_{xy}(#eta) w.r.t 0");
    g_KS_dxy_eta_vs_run[j]->GetYaxis()->SetRangeUser(-20., 1.);
    beautify(g_KS_dxy_eta_vs_run[j]);
    //g_KS_dxy_eta_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_KS_dxy_eta_vs_run[j]->Draw("AP");
    } else {
      g_KS_dxy_eta_vs_run[j]->Draw("Psame");
    }

    if (time_axis_format) {
      timify(g_KS_dxy_eta_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    //3d pad
    c_KSScore_vs_run->cd(3);
    adjustmargins(c_KSScore_vs_run->cd(3));
    g_KS_dz_phi_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_KS_dz_phi_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_KS_dz_phi_vs_run[j]->SetLineColor(pv::colors[j]);

    g_KS_dz_phi_vs_run[j]->SetName(Form("g_KS_dz_phi_%s", LegLabels[j].Data()));
    g_KS_dz_phi_vs_run[j]->SetTitle(Form("log_{10}(KS-score) of d_{z}(#varphi) vs %s", theType.Data()));
    g_KS_dz_phi_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_KS_dz_phi_vs_run[j]->GetYaxis()->SetTitle("log_{10}(KS-score) of d_{z}(#phi) w.r.t 0");
    g_KS_dz_phi_vs_run[j]->GetYaxis()->SetRangeUser(-20., 1.);
    beautify(g_KS_dz_phi_vs_run[j]);
    //g_KS_dz_phi_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_KS_dz_phi_vs_run[j]->Draw("AP");
    } else {
      g_KS_dz_phi_vs_run[j]->Draw("Psame");
    }

    if (time_axis_format) {
      timify(g_KS_dz_phi_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }

    //4th pad
    c_KSScore_vs_run->cd(4);
    adjustmargins(c_KSScore_vs_run->cd(4));
    g_KS_dz_eta_vs_run[j]->SetMarkerStyle(pv::markers[j]);
    g_KS_dz_eta_vs_run[j]->SetMarkerColor(pv::colors[j]);
    g_KS_dz_eta_vs_run[j]->SetLineColor(pv::colors[j]);

    g_KS_dz_eta_vs_run[j]->SetName(Form("g_KS_dz_eta_%s", LegLabels[j].Data()));
    g_KS_dz_eta_vs_run[j]->SetTitle(Form("log_{10}(KS-score) of d_{z}(#eta) vs %s", theType.Data()));
    g_KS_dz_eta_vs_run[j]->GetXaxis()->SetTitle(theTypeLabel.Data());
    g_KS_dz_eta_vs_run[j]->GetYaxis()->SetTitle("log_{10}(KS-score) of d_{z}(#eta) w.r.t 0");
    g_KS_dz_eta_vs_run[j]->GetYaxis()->SetRangeUser(-20., 1.);
    beautify(g_KS_dz_eta_vs_run[j]);
    //g_KS_dz_eta_vs_run[j]->GetYaxis()->SetTitleOffset(1.3);

    if (j == 0) {
      g_KS_dz_eta_vs_run[j]->Draw("AP");
    } else {
      g_KS_dz_eta_vs_run[j]->Draw("Psame");
    }

    if (time_axis_format) {
      timify(g_KS_dz_eta_vs_run[j]);
    }

    if (j == nDirs_ - 1) {
      my_lego->Draw("same");
    }
  }

  // delete the array for the maxima
  delete arr_dxy_phi;
  delete arr_dz_phi;
  delete arr_dxy_eta;
  delete arr_dz_eta;

  TString append;
  if (lumi_axis_format) {
    append = "lumi";
  } else {
    if (time_axis_format) {
      append = "date";
    } else {
      append = "run";
    }
  }

  c_dxy_phi_vs_run->SaveAs("dxy_phi_vs_" + append + ".pdf");
  c_dxy_phi_vs_run->SaveAs("dxy_phi_vs_" + append + ".png");
  c_dxy_phi_vs_run->SaveAs("dxy_phi_vs_" + append + ".eps");

  c_dxy_eta_vs_run->SaveAs("dxy_eta_vs_" + append + ".pdf");
  c_dxy_eta_vs_run->SaveAs("dxy_eta_vs_" + append + ".png");
  c_dxy_eta_vs_run->SaveAs("dxy_eta_vs_" + append + ".eps");

  c_dz_phi_vs_run->SaveAs("dz_phi_vs_" + append + ".pdf");
  c_dz_phi_vs_run->SaveAs("dz_phi_vs_" + append + ".png");
  c_dz_phi_vs_run->SaveAs("dz_phi_vs_" + append + ".eps");

  c_dz_eta_vs_run->SaveAs("dz_eta_vs_" + append + ".pdf");
  c_dz_eta_vs_run->SaveAs("dz_eta_vs_" + append + ".png");
  c_dz_eta_vs_run->SaveAs("dz_eta_vs_" + append + ".eps");

  TCanvas dummyC("dummyC", "dummyC", 2000, 800);
  dummyC.Print("combined.pdf[");
  c_dxy_phi_vs_run->Print("combined.pdf");
  c_dxy_eta_vs_run->Print("combined.pdf");
  c_dz_phi_vs_run->Print("combined.pdf");
  c_dz_eta_vs_run->Print("combined.pdf");
  dummyC.Print("combined.pdf]");

  // mean

  c_mean_dxy_phi_vs_run->SaveAs("mean_dxy_phi_vs_" + append + ".pdf");
  c_mean_dxy_phi_vs_run->SaveAs("mean_dxy_phi_vs_" + append + ".png");

  c_mean_dxy_eta_vs_run->SaveAs("mean_dxy_eta_vs_" + append + ".pdf");
  c_mean_dxy_eta_vs_run->SaveAs("mean_dxy_eta_vs_" + append + ".png");

  c_mean_dz_phi_vs_run->SaveAs("mean_dz_phi_vs_" + append + ".pdf");
  c_mean_dz_phi_vs_run->SaveAs("mean_dz_phi_vs_" + append + ".png");

  c_mean_dz_eta_vs_run->SaveAs("mean_dz_eta_vs_" + append + ".pdf");
  c_mean_dz_eta_vs_run->SaveAs("mean_dz_eta_vs_" + append + ".png");

  TCanvas dummyC2("dummyC2", "dummyC2", 2000, 800);
  dummyC2.Print("means.pdf[");
  c_mean_dxy_phi_vs_run->Print("means.pdf");
  c_mean_dxy_eta_vs_run->Print("means.pdf");
  c_mean_dz_phi_vs_run->Print("means.pdf");
  c_mean_dz_eta_vs_run->Print("means.pdf");
  dummyC2.Print("means.pdf]");

  // RMS

  c_RMS_dxy_phi_vs_run->SaveAs("RMS_dxy_phi_vs_" + append + ".pdf");
  c_RMS_dxy_phi_vs_run->SaveAs("RMS_dxy_phi_vs_" + append + ".png");

  c_RMS_dxy_eta_vs_run->SaveAs("RMS_dxy_eta_vs_" + append + ".pdf");
  c_RMS_dxy_eta_vs_run->SaveAs("RMS_dxy_eta_vs_" + append + ".png");

  c_RMS_dz_phi_vs_run->SaveAs("RMS_dz_phi_vs_" + append + ".pdf");
  c_RMS_dz_phi_vs_run->SaveAs("RMS_dz_phi_vs_" + append + ".png");

  c_RMS_dz_eta_vs_run->SaveAs("RMS_dz_eta_vs_" + append + ".pdf");
  c_RMS_dz_eta_vs_run->SaveAs("RMS_dz_eta_vs_" + append + ".png");

  TCanvas dummyC3("dummyC3", "dummyC3", 2000, 800);
  dummyC3.Print("RMSs.pdf[");
  c_RMS_dxy_phi_vs_run->Print("RMSs.pdf");
  c_RMS_dxy_eta_vs_run->Print("RMSs.pdf");
  c_RMS_dz_phi_vs_run->Print("RMSs.pdf");
  c_RMS_dz_eta_vs_run->Print("RMSs.pdf");
  dummyC3.Print("RMSs.pdf]");

  // scatter

  Scatter_dxy_vs_run->SaveAs("Scatter_dxy_vs_" + append + ".pdf");
  Scatter_dxy_vs_run->SaveAs("Scatter_dxy_vs_" + append + ".png");

  Scatter_dz_vs_run->SaveAs("Scatter_dz_vs_" + append + ".pdf");
  Scatter_dz_vs_run->SaveAs("Scatter_dz_vs_" + append + ".png");

  // chi2

  c_chisquare_vs_run->SaveAs("chi2pol0fit_vs_" + append + ".pdf");
  c_chisquare_vs_run->SaveAs("chi2pol0fit_vs_" + append + ".png");

  // KS score

  c_KSScore_vs_run->SaveAs("KSScore_vs_" + append + ".pdf");
  c_KSScore_vs_run->SaveAs("KSScore_vs_" + append + ".png");

  // do all the deletes

  for (int iDir = 0; iDir < nDirs_; iDir++) {
    delete g_dxy_phi_vs_run[iDir];
    delete g_chi2_dxy_phi_vs_run[iDir];
    delete g_KS_dxy_phi_vs_run[iDir];
    delete g_dxy_phi_hi_vs_run[iDir];
    delete g_dxy_phi_lo_vs_run[iDir];

    delete g_dxy_eta_vs_run[iDir];
    delete g_chi2_dxy_eta_vs_run[iDir];
    delete g_KS_dxy_eta_vs_run[iDir];
    delete g_dxy_eta_hi_vs_run[iDir];
    delete g_dxy_eta_lo_vs_run[iDir];

    delete g_dz_phi_vs_run[iDir];
    delete g_chi2_dz_phi_vs_run[iDir];
    delete g_KS_dz_phi_vs_run[iDir];
    delete g_dz_phi_hi_vs_run[iDir];
    delete g_dz_phi_lo_vs_run[iDir];

    delete g_dz_eta_vs_run[iDir];
    delete g_chi2_dz_eta_vs_run[iDir];
    delete g_KS_dz_eta_vs_run[iDir];
    delete g_dz_eta_hi_vs_run[iDir];
    delete g_dz_eta_lo_vs_run[iDir];

    delete h_RMS_dxy_phi_vs_run[iDir];
    delete h_RMS_dxy_eta_vs_run[iDir];
    delete h_RMS_dz_phi_vs_run[iDir];
    delete h_RMS_dz_eta_vs_run[iDir];
  }

  // mv the run-by-run plots into the folders

  gSystem->mkdir("Biases");
  TString processline = ".! mv Bias*.p* ./Biases/";
  std::cout << "Executing: \n" << processline << "\n" << std::endl;
  gROOT->ProcessLine(processline.Data());
  gSystem->Sleep(100);
  processline.Clear();

  gSystem->mkdir("ResolutionsVsPt");
  processline = ".! mv ResolutionsVsPt*.p* ./ResolutionsVsPt/";
  std::cout << "Executing: \n" << processline << "\n" << std::endl;
  gROOT->ProcessLine(processline.Data());
  gSystem->Sleep(100);
  processline.Clear();

  gSystem->mkdir("Resolutions");
  processline = ".! mv Resolutions*.p* ./Resolutions/";
  std::cout << "Executing: \n" << processline << "\n" << std::endl;
  gROOT->ProcessLine(processline.Data());
  gSystem->Sleep(100);
  processline.Clear();

  gSystem->mkdir("Pulls");
  processline = ".! mv Pulls*.p* ./Pulls/";
  std::cout << "Executing: \n" << processline << "\n" << std::endl;
  gROOT->ProcessLine(processline.Data());
  gSystem->Sleep(100);
  processline.Clear();

  timer.Stop();
  timer.Print();
}

/*! \fn outputGraphs
 *  \brief function to build the output graphs
 */

/*--------------------------------------------------------------------*/
void outputGraphs(const pv::wrappedTrends &allInputs,
                  const std::vector<double> &ticks,
                  const std::vector<double> &ex_ticks,
                  TCanvas *&canv,
                  TCanvas *&mean_canv,
                  TCanvas *&rms_canv,
                  TGraph *&g_mean,
                  TGraph *&g_chi2,
                  TGraph *&g_KS,
                  TGraph *&g_low,
                  TGraph *&g_high,
                  TGraphAsymmErrors *&g_asym,
                  TH1F *h_RMS[],
                  const pv::bundle &mybundle,
                  const pv::view &theView,
                  const int index,
                  TObjArray *&array,
                  const TString &label,
                  TLegend *&legend)
/*--------------------------------------------------------------------*/
{
  g_mean = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getMean()[label])[0]));
  g_chi2 = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getChi2()[label])[0]));
  g_KS = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getKS()[label])[0]));
  g_high = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getHigh()[label])[0]));
  g_low = new TGraph(ticks.size(), &(ticks[0]), &((allInputs.getLow()[label])[0]));

  g_asym = new TGraphAsymmErrors(ticks.size(),
                                 &(ticks[0]),
                                 &((allInputs.getMean()[label])[0]),
                                 &(ex_ticks[0]),
                                 &(ex_ticks[0]),
                                 &((allInputs.getLowErr()[label])[0]),
                                 &((allInputs.getHighErr()[label])[0]));

  adjustmargins(canv);
  canv->cd();
  g_asym->SetFillStyle(3005);
  g_asym->SetFillColor(pv::colors[index]);
  g_mean->SetMarkerStyle(pv::markers[index]);
  g_mean->SetMarkerColor(pv::colors[index]);
  g_mean->SetLineColor(pv::colors[index]);
  g_mean->SetMarkerSize(1.5);
  g_high->SetLineColor(pv::colors[index]);
  g_low->SetLineColor(pv::colors[index]);
  beautify(g_mean);
  beautify(g_asym);

  if (theView == pv::dxyphi) {
    legend->AddEntry(g_mean, label, "PL");
  }

  const char *coord;
  const char *kin;
  float ampl;

  switch (theView) {
    case pv::dxyphi:
      coord = "xy";
      kin = "phi";
      ampl = 40;
      break;
    case pv::dzphi:
      coord = "z";
      kin = "phi";
      ampl = 40;
      break;
    case pv::dxyeta:
      coord = "xy";
      kin = "eta";
      ampl = 40;
      break;
    case pv::dzeta:
      coord = "z";
      kin = "eta";
      ampl = 60;
      break;
    default:
      coord = "unknown";
      kin = "unknown";
      break;
  }

  g_mean->SetName(Form("g_bias_d%s_%s_%s", coord, kin, label.Data()));
  g_mean->SetTitle(Form("Bias of d_{%s}(#%s) vs %s", coord, kin, mybundle.getDataType()));
  g_mean->GetXaxis()->SetTitle(mybundle.getDataTypeLabel());
  g_mean->GetYaxis()->SetTitle(Form("#LT d_{%s}(#%s) #GT [#mum]", coord, kin));
  g_mean->GetYaxis()->SetRangeUser(-ampl, ampl);

  std::cout << "===================================================================================================="
            << std::endl;
  std::cout << mybundle.getTotalLumi() << std::endl;
  std::cout << "===================================================================================================="
            << std::endl;

  g_asym->SetName(Form("gerr_bias_d%s_%s_%s", coord, kin, label.Data()));
  g_asym->SetTitle(Form("Bias of d_{%s}(#%s) vs %s", coord, kin, mybundle.getDataType()));
  g_asym->GetXaxis()->SetTitle(mybundle.getDataTypeLabel());
  g_asym->GetYaxis()->SetTitle(Form("#LT d_{%s}(#%s) #GT [#mum]", coord, kin));
  g_asym->GetYaxis()->SetRangeUser(-ampl, ampl);

  //g_mean->GetXaxis()->UnZoom();
  //g_asym->GetXaxis()->UnZoom();

  if (index == 0) {
    g_asym->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
    g_mean->Draw("AP");
    g_asym->Draw("P3same");
  } else {
    g_mean->Draw("Psame");
    g_asym->Draw("3same");
  }
  g_high->Draw("Lsame");
  g_low->Draw("Lsame");

  if (mybundle.isTimeAxis()) {
    timify(g_asym);
    timify(g_mean);
    timify(g_high);
    timify(g_low);
  }

  if (mybundle.isLumiAxis()) {
    g_asym->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
    g_mean->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
    g_high->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
    g_low->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
  }

  if (index == (mybundle.getNObjects() - 1)) {
    legend->Draw("same");
    TH1F *theZero = DrawConstantGraph(g_mean, 1, 0.);
    theZero->Draw("E1same][");
  }

  auto current_pad = static_cast<TPad *>(gPad);
  cmsPrel(current_pad);

  superImposeIOVBoundaries(
      canv, mybundle.isLumiAxis(), mybundle.isTimeAxis(), mybundle.getLumiMapByRun(), mybundle.getTimes());

  // mean only
  adjustmargins(mean_canv);
  mean_canv->cd();
  auto gprime = (TGraph *)g_mean->Clone();
  if (index == 0) {
    gprime->GetYaxis()->SetRangeUser(-10., 10.);
    if (mybundle.isLumiAxis())
      gprime->GetXaxis()->SetRangeUser(0., mybundle.getTotalLumi());
    gprime->Draw("AP");
  } else {
    gprime->Draw("Psame");
  }

  if (index == (mybundle.getNObjects() - 1)) {
    legend->Draw("same");
  }

  if (index == 0) {
    TH1F *theZero = DrawConstantGraph(gprime, 2, 0.);
    theZero->Draw("E1same][");

    auto current_pad = static_cast<TPad *>(gPad);
    cmsPrel(current_pad);

    superImposeIOVBoundaries(
        mean_canv, mybundle.isLumiAxis(), mybundle.isTimeAxis(), mybundle.getLumiMapByRun(), mybundle.getTimes());
  }

  // scatter or RMS TH1
  h_RMS[index] = new TH1F(Form("h_RMS_dz_eta_%s", label.Data()),
                          Form("scatter of d_{%s}(#%s) vs %s;%s;%s of d_{%s}(#%s) [#mum]",
                               coord,
                               kin,
                               mybundle.getDataType(),
                               mybundle.getDataTypeLabel(),
                               (mybundle.isUsingRMS() ? "RMS" : "peak-to-peak deviation"),
                               coord,
                               kin),
                          ticks.size() - 1,
                          &(ticks[0]));
  h_RMS[index]->SetStats(kFALSE);

  int bincounter = 0;
  for (const auto &tick : ticks) {
    bincounter++;
    h_RMS[index]->SetBinContent(
        bincounter, std::abs(allInputs.getHigh()[label][bincounter - 1] - allInputs.getLow()[label][bincounter - 1]));
    h_RMS[index]->SetBinError(bincounter, 0.01);
  }

  h_RMS[index]->SetLineColor(pv::colors[index]);
  h_RMS[index]->SetLineWidth(2);
  h_RMS[index]->SetMarkerSize(1.5);
  h_RMS[index]->SetMarkerStyle(pv::markers[index]);
  h_RMS[index]->SetMarkerColor(pv::colors[index]);
  adjustmargins(rms_canv);
  rms_canv->cd();
  beautify(h_RMS[index]);

  if (mybundle.isTimeAxis()) {
    timify(h_RMS[index]);
  }

  array->Add(h_RMS[index]);

  // at the last file re-loop
  if (index == (mybundle.getNObjects() - 1)) {
    auto theMax = getMaximumFromArray(array);
    std::cout << "the max for d" << coord << "(" << kin << ") RMS is " << theMax << std::endl;

    for (Int_t k = 0; k < mybundle.getNObjects(); k++) {
      //h_RMS[k]->GetYaxis()->SetRangeUser(-theMax*0.45,theMax*1.40);
      h_RMS[k]->GetYaxis()->SetRangeUser(0., theMax * 1.80);
      if (k == 0) {
        h_RMS[k]->Draw("L");
      } else {
        h_RMS[k]->Draw("Lsame");
      }
    }
    legend->Draw("same");
    TH1F *theConst = DrawConstant(h_RMS[index], 1, 0.);
    //theConst->Draw("same][");

    current_pad = static_cast<TPad *>(gPad);
    cmsPrel(current_pad);

    superImposeIOVBoundaries(
        rms_canv, mybundle.isLumiAxis(), mybundle.isTimeAxis(), mybundle.getLumiMapByRun(), mybundle.getTimes());
  }
}

/*! \fn list_files
 *  \brief utility function to list of filles in a directory
 */

/*--------------------------------------------------------------------*/
std::vector<int> list_files(const char *dirname, const char *ext)
/*--------------------------------------------------------------------*/
{
  std::vector<int> theRunNumbers;

  TSystemDirectory dir(dirname, dirname);
  TList *files = dir.GetListOfFiles();
  if (files) {
    TSystemFile *file;
    TString fname;
    TIter next(files);
    while ((file = (TSystemFile *)next())) {
      fname = file->GetName();
      if (!file->IsDirectory() && fname.EndsWith(ext) && fname.BeginsWith("PVValidation")) {
        //std::cout << fname.Data() << std::endl;
        TObjArray *bits = fname.Tokenize("_");
        TString theRun = bits->At(2)->GetName();
        //std::cout << theRun << std::endl;
        TString formatRun = (theRun.ReplaceAll(".root", "")).ReplaceAll("_", "");
        //std::cout << dirname << " "<< formatRun.Atoi() << std::endl;
        theRunNumbers.push_back(formatRun.Atoi());
      }
    }
  }
  return theRunNumbers;
}

/*! \fn arrangeOutCanvas
 *  \brief utility function to arrange the plots per run nicely in a TCanvas
 */

/*--------------------------------------------------------------------*/
void arrangeOutCanvas(TCanvas *canv,
                      TH1F *m_11Trend[100],
                      TH1F *m_12Trend[100],
                      TH1F *m_21Trend[100],
                      TH1F *m_22Trend[100],
                      Int_t nDirs,
                      TString LegLabels[10],
                      unsigned int theRun) {
  /*--------------------------------------------------------------------*/

  TLegend *lego = new TLegend(0.19, 0.80, 0.79, 0.93);
  //lego-> SetNColumns(2);
  lego->SetFillColor(10);
  lego->SetTextSize(0.042);
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *ptDate = new TPaveText(0.19, 0.95, 0.45, 0.99, "blNDC");
  ptDate->SetFillColor(kYellow);
  //ptDate->SetFillColor(10);
  ptDate->SetBorderSize(1);
  ptDate->SetLineColor(kBlue);
  ptDate->SetLineWidth(1);
  ptDate->SetTextFont(42);
  TText *textDate = ptDate->AddText(Form("Run: %i", theRun));
  textDate->SetTextSize(0.04);
  textDate->SetTextColor(kBlue);
  textDate->SetTextAlign(22);

  canv->SetFillColor(10);
  canv->Divide(2, 2);

  TH1F *dBiasTrend[4][nDirs];

  for (Int_t i = 0; i < nDirs; i++) {
    dBiasTrend[0][i] = m_11Trend[i];
    dBiasTrend[1][i] = m_12Trend[i];
    dBiasTrend[2][i] = m_21Trend[i];
    dBiasTrend[3][i] = m_22Trend[i];
  }

  Double_t absmin[4] = {999., 999., 999., 999.};
  Double_t absmax[4] = {-999., -999. - 999., -999.};

  for (Int_t k = 0; k < 4; k++) {
    canv->cd(k + 1)->SetBottomMargin(0.14);
    canv->cd(k + 1)->SetLeftMargin(0.18);
    canv->cd(k + 1)->SetRightMargin(0.01);
    canv->cd(k + 1)->SetTopMargin(0.06);

    canv->cd(k + 1);

    for (Int_t i = 0; i < nDirs; i++) {
      if (dBiasTrend[k][i]->GetMaximum() > absmax[k])
        absmax[k] = dBiasTrend[k][i]->GetMaximum();
      if (dBiasTrend[k][i]->GetMinimum() < absmin[k])
        absmin[k] = dBiasTrend[k][i]->GetMinimum();
    }

    Double_t safeDelta = (absmax[k] - absmin[k]) / 8.;
    Double_t theExtreme = std::max(absmax[k], TMath::Abs(absmin[k]));

    for (Int_t i = 0; i < nDirs; i++) {
      if (i == 0) {
        TString theTitle = dBiasTrend[k][i]->GetName();

        if (theTitle.Contains("norm")) {
          //dBiasTrend[k][i]->GetYaxis()->SetRangeUser(std::min(-0.48,absmin[k]-safeDelta/2.),std::max(0.48,absmax[k]+safeDelta/2.));
          dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 1.8);
        } else {
          if (!theTitle.Contains("width")) {
            if (theTitle.Contains("ladder") || theTitle.Contains("modZ")) {
              dBiasTrend[k][i]->GetYaxis()->SetRangeUser(std::min(-40., -theExtreme - (safeDelta / 2.)),
                                                         std::max(40., theExtreme + (safeDelta / 2.)));
            } else {
              dBiasTrend[k][i]->GetYaxis()->SetRangeUser(-40., 40.);
            }
          } else {
            // dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0.,theExtreme+(safeDelta/2.));
            // if(theTitle.Contains("eta")) {
            //   dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0.,500.);
            // } else {
            //   dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0.,200.);
            // }
            auto my_view = checkTheView(theTitle);

            //std::cout<<" ----------------------------------> " << theTitle << " view: " << my_view <<  std::endl;

            switch (my_view) {
              case pv::dxyphi:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 200.);
                break;
              case pv::dzphi:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 400.);
                break;
              case pv::dxyeta:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 300.);
                break;
              case pv::dzeta:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 1.e3);
                break;
              case pv::pT:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 200.);
                break;
              case pv::generic:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 350.);
                break;
              default:
                dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., 300.);
            }
          }
        }

        dBiasTrend[k][i]->Draw("Le1");
        makeNewXAxis(dBiasTrend[k][i]);

        Double_t theC = -1.;

        if (theTitle.Contains("width")) {
          if (theTitle.Contains("norm")) {
            theC = 1.;
          } else {
            theC = -1.;
          }
        } else {
          theC = 0.;
        }

        TH1F *theConst = DrawConstant(dBiasTrend[k][i], 1, theC);
        theConst->Draw("PLsame][");

      } else {
        dBiasTrend[k][i]->Draw("Le1sames");
        makeNewXAxis(dBiasTrend[k][i]);
      }
      TPad *current_pad = static_cast<TPad *>(canv->GetPad(k + 1));
      cmsPrel(current_pad, 2);
      ptDate->Draw("same");

      if (k == 0) {
        lego->AddEntry(dBiasTrend[k][i], LegLabels[i]);
      }
    }

    lego->Draw();
  }
}

/*! \fn MakeNiceTrendPlotStype
 *  \brief utility function to embellish trend plot style
 */

/*--------------------------------------------------------------------*/
void MakeNiceTrendPlotStyle(TH1 *hist, Int_t color)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.065);
  hist->GetYaxis()->SetTitleSize(0.065);
  hist->GetXaxis()->SetTitleOffset(1.0);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.07);
  //hist->GetXaxis()->SetNdivisions(505);
  if (color != 8) {
    hist->SetMarkerSize(1.5);
  } else {
    hist->SetLineWidth(3);
    hist->SetMarkerSize(0.0);
  }
  hist->SetMarkerStyle(pv::markers[color]);
  hist->SetLineColor(pv::colors[color]);
  hist->SetMarkerColor(pv::colors[color]);
}

/*! \fn maxkeNewAxis
 *  \brief utility function to re-make x-axis with correct binning
 */

/*--------------------------------------------------------------------*/
void makeNewXAxis(TH1 *h)
/*--------------------------------------------------------------------*/
{
  TString myTitle = h->GetName();
  float axmin = -999;
  float axmax = 999.;
  int ndiv = 510;
  if (myTitle.Contains("eta")) {
    axmin = -2.7;
    axmax = 2.7;
    ndiv = 505;
  } else if (myTitle.Contains("phi")) {
    axmin = -TMath::Pi();
    axmax = TMath::Pi();
    ndiv = 510;
  } else if (myTitle.Contains("pT")) {
    axmin = 0;
    axmax = 19.99;
    ndiv = 510;
  } else if (myTitle.Contains("ladder")) {
    axmin = 0;
    axmax = 12;
    ndiv = 510;
  } else if (myTitle.Contains("modZ")) {
    axmin = 0;
    axmax = 8;
    ndiv = 510;
  } else {
    std::cout << "unrecognized variable" << std::endl;
  }

  // Remove the current axis
  h->GetXaxis()->SetLabelOffset(999);
  h->GetXaxis()->SetTickLength(0);

  if (myTitle.Contains("phi")) {
    h->GetXaxis()->SetTitle("#varphi (sector) [rad]");
  }

  // Redraw the new axis
  gPad->Update();

  TGaxis *newaxis =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymin(), axmin, axmax, ndiv, "SDH");

  TGaxis *newaxisup =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymax(), gPad->GetUxmax(), gPad->GetUymax(), axmin, axmax, ndiv, "-SDH");

  newaxis->SetLabelOffset(0.02);
  newaxis->SetLabelFont(42);
  newaxis->SetLabelSize(0.05);

  newaxisup->SetLabelOffset(-0.02);
  newaxisup->SetLabelFont(42);
  newaxisup->SetLabelSize(0);

  newaxis->Draw();
  newaxisup->Draw();
}

/*! \fn setStyle
 *  \brief main function to set the plotting style
 */

/*--------------------------------------------------------------------*/
void setStyle() {
  /*--------------------------------------------------------------------*/

  TGaxis::SetMaxDigits(6);

  TH1::StatOverflows(kTRUE);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("e");
  //gStyle->SetPadTopMargin(0.05);
  //gStyle->SetPadBottomMargin(0.15);
  //gStyle->SetPadLeftMargin(0.17);
  //gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadBorderMode(0);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleColor(1);
  gStyle->SetTitleTextColor(1);
  gStyle->SetTitleFontSize(0.06);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatFont(42);
  gStyle->SetStatFontSize(0.05);  ///---> gStyle->SetStatFontSize(0.025);
  gStyle->SetStatTextColor(1);
  gStyle->SetStatFormat("6.4g");
  gStyle->SetStatBorderSize(1);
  gStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  gStyle->SetPadTickY(1);
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptFit(1);
  gStyle->SetNdivisions(510);

  //gStyle->SetPalette(kInvertedDarkBodyRadiator);

  const Int_t NRGBs = 5;
  const Int_t NCont = 255;

  /*
  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  */

  Double_t stops[NRGBs] = {0.00, 0.01, 0.05, 0.09, 0.1};
  Double_t red[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
  Double_t green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
  Double_t blue[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);
}

// /*--------------------------------------------------------------------*/
// void cmsPrel(TPad* pad) {
// /*--------------------------------------------------------------------*/

//   float H = pad->GetWh();
//   float W = pad->GetWw();
//   float l = pad->GetLeftMargin();
//   float t = pad->GetTopMargin();
//   float r = pad->GetRightMargin();
//   float b = pad->GetBottomMargin();
//   float relPosX = 0.009;
//   float relPosY = 0.045;
//   float lumiTextOffset = 0.8;

//   TLatex *latex = new TLatex();
//   latex->SetNDC();
//   latex->SetTextSize(0.045);

//   float posX_    = 1-r - relPosX*(1-l-r);
//   float posXCMS_ = posX_- 0.125;
//   float posY_ =  1-t + 0.05; /// - relPosY*(1-t-b);
//   float factor = 1./0.86;

//   latex->SetTextAlign(33);
//   latex->SetTextFont(61);
//   latex->SetTextSize(0.045*factor);
//   latex->DrawLatex(posXCMS_,posY_,"CMS");
//   latex->SetTextSize(0.045);
//   latex->SetTextFont(42); //22
//   latex->DrawLatex(posX_,posY_,"Internal (13 TeV)");
//   //latex->DrawLatex(posX_,posY_,"CMS Preliminary (13 TeV)");
//   //latex->DrawLatex(posX_,posY_,"CMS 2017 Work in progress (13 TeV)");

// }

/*! \fn cmsPrel
 *  \brief utility function to put the CMS paraphernalia on canvas
 */

/*--------------------------------------------------------------------*/
void cmsPrel(TPad *pad, size_t ipads) {
  /*--------------------------------------------------------------------*/

  float H = pad->GetWh();
  float W = pad->GetWw();
  float l = pad->GetLeftMargin();
  float t = pad->GetTopMargin();
  float r = pad->GetRightMargin();
  float b = pad->GetBottomMargin();
  float relPosX = 0.009;
  float relPosY = 0.045;
  float lumiTextOffset = 0.8;

  TLatex *latex = new TLatex();
  latex->SetNDC();
  latex->SetTextSize(0.045);

  float posX_ = 1 - (r / ipads);
  float posY_ = 1 - t + 0.05;  /// - relPosY*(1-t-b);
  float factor = 1. / 0.82;

  latex->SetTextAlign(33);
  latex->SetTextSize(0.045);
  latex->SetTextFont(42);  //22
  latex->DrawLatex(posX_, posY_, "Internal (13 TeV)");

  UInt_t w;
  UInt_t h;
  latex->GetTextExtent(w, h, "Internal (13 TeV)");
  float size = w / (W / ipads);
  //std::cout<<w<<" "<<" "<<W<<" "<<size<<std::endl;
  float posXCMS_ = posX_ - size * (1 + 0.025 * ipads);

  latex->SetTextAlign(33);
  latex->SetTextFont(61);
  latex->SetTextSize(0.045 * factor);
  latex->DrawLatex(posXCMS_, posY_ + 0.004, "CMS");

  //latex->DrawLatex(posX_,posY_,"CMS Preliminary (13 TeV)");
  //latex->DrawLatex(posX_,posY_,"CMS 2017 Work in progress (13 TeV)");
}

/*! \fn DrawConstant
 *  \brief utility function to draw a constant histogram
 */

/*--------------------------------------------------------------------*/
TH1F *DrawConstant(TH1F *hist, Int_t iter, Double_t theConst)
/*--------------------------------------------------------------------*/
{
  Int_t nbins = hist->GetNbinsX();
  Double_t lowedge = hist->GetBinLowEdge(1);
  Double_t highedge = hist->GetBinLowEdge(nbins + 1);

  TH1F *hzero = new TH1F(Form("hconst_%s_%i", hist->GetName(), iter),
                         Form("hconst_%s_%i", hist->GetName(), iter),
                         nbins,
                         lowedge,
                         highedge);
  for (Int_t i = 0; i <= hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, theConst);
    hzero->SetBinError(i, 0.);
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*! \fn DrawConstant
 *  \brief utility function to draw a constant histogram with erros !=0
 */

/*--------------------------------------------------------------------*/
TH1F *DrawConstantWithErr(TH1F *hist, Int_t iter, Double_t theConst)
/*--------------------------------------------------------------------*/
{
  Int_t nbins = hist->GetNbinsX();
  Double_t lowedge = hist->GetBinLowEdge(1);
  Double_t highedge = hist->GetBinLowEdge(nbins + 1);

  TH1F *hzero = new TH1F(Form("hconst_%s_%i", hist->GetName(), iter),
                         Form("hconst_%s_%i", hist->GetName(), iter),
                         nbins,
                         lowedge,
                         highedge);
  for (Int_t i = 0; i <= hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, theConst);
    hzero->SetBinError(i, hist->GetBinError(i));
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*! \fn DrawConstantGraph
 *  \brief utility function to draw a constant TGraph
 */

/*--------------------------------------------------------------------*/
TH1F *DrawConstantGraph(TGraph *graph, Int_t iter, Double_t theConst)
/*--------------------------------------------------------------------*/
{
  Double_t xmin = graph->GetXaxis()->GetXmin();  //TMath::MinElement(graph->GetN(),graph->GetX());
  Double_t xmax = graph->GetXaxis()->GetXmax();  //TMath::MaxElement(graph->GetN(),graph->GetX());

  //std::cout<<xmin<<" : "<<xmax<<std::endl;

  TH1F *hzero = new TH1F(Form("hconst_%s_%i", graph->GetName(), iter),
                         Form("hconst_%s_%i", graph->GetName(), iter),
                         graph->GetN(),
                         xmin,
                         xmax);
  for (Int_t i = 0; i <= hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, theConst);
    hzero->SetBinError(i, 0.);
  }

  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*! \fn getUnrolledHisto
 *  \brief utility function to tranform a TH1 into a vector of floats
 */

/*--------------------------------------------------------------------*/
unrolledHisto getUnrolledHisto(TH1F *hist)
/*--------------------------------------------------------------------*/
{
  /*
    Double_t y_min = hist->GetBinLowEdge(1);
    Double_t y_max = hist->GetBinLowEdge(hist->GetNbinsX()+1);
  */

  Double_t y_min = -0.1;
  Double_t y_max = 0.1;

  std::vector<Double_t> contents;
  for (int j = 0; j < hist->GetNbinsX(); j++) {
    if (std::abs(hist->GetXaxis()->GetBinCenter(j)) <= 0.1)
      contents.push_back(hist->GetBinContent(j + 1));
  }

  auto ret = unrolledHisto(y_min, y_max, contents.size(), contents);
  return ret;
}

/*! \fn getBiases
 *  \brief utility function to extract characterization of the PV bias plot
 */

/*--------------------------------------------------------------------*/
pv::biases getBiases(TH1F *hist)
/*--------------------------------------------------------------------*/
{
  int nbins = hist->GetNbinsX();

  //extract median from histogram
  double *y = new double[nbins];
  double *err = new double[nbins];

  // remember for weight means <x> = sum_i (x_i* w_i) / sum_i w_i ; where w_i = 1/sigma^2_i

  for (int j = 0; j < nbins; j++) {
    y[j] = hist->GetBinContent(j + 1);
    if (hist->GetBinError(j + 1) != 0.) {
      err[j] = 1. / (hist->GetBinError(j + 1) * hist->GetBinError(j + 1));
    } else {
      err[j] = 0.;
    }
  }

  Double_t w_mean = TMath::Mean(nbins, y, err);
  Double_t w_rms = TMath::RMS(nbins, y, err);

  Double_t mean = TMath::Mean(nbins, y);
  Double_t rms = TMath::RMS(nbins, y);

  Double_t max = hist->GetMaximum();
  Double_t min = hist->GetMinimum();

  // in case one would like to use a pol0 fit
  hist->Fit("pol0", "Q0+");
  TF1 *f = (TF1 *)hist->FindObject("pol0");
  //f->SetLineColor(hist->GetLineColor());
  //f->SetLineStyle(hist->GetLineStyle());
  Double_t chi2 = f->GetChisquare();
  Int_t ndf = f->GetNDF();

  TH1F *theZero = DrawConstantWithErr(hist, 1, 1.);
  TH1F *displaced = (TH1F *)hist->Clone("displaced");
  displaced->Add(theZero);
  Double_t ksScore = std::max(-20., TMath::Log10(displaced->KolmogorovTest(theZero)));
  Double_t chi2Score = displaced->Chi2Test(theZero);

  /*
    std::pair<std::pair<Double_t,Double_t>, Double_t> result;
    std::pair<Double_t,Double_t> resultBounds;
    resultBounds = useRMS_ ? std::make_pair(mean-rms,mean+rms) :  std::make_pair(min,max)  ;
    result = make_pair(resultBounds,mean);
  */

  pv::biases result(mean, rms, w_mean, w_rms, min, max, chi2, ndf, ksScore);

  delete theZero;
  delete displaced;
  return result;
}

/*! \fn beautify
 *  \brief utility function to beautify a TGraph
 */

/*--------------------------------------------------------------------*/
void beautify(TGraph *g) {
  /*--------------------------------------------------------------------*/
  g->GetXaxis()->SetLabelFont(42);
  g->GetYaxis()->SetLabelFont(42);
  g->GetYaxis()->SetLabelSize(.055);
  g->GetXaxis()->SetLabelSize(.055);
  g->GetYaxis()->SetTitleSize(.055);
  g->GetXaxis()->SetTitleSize(.055);
  g->GetXaxis()->SetTitleOffset(1.1);
  g->GetYaxis()->SetTitleOffset(0.6);
  g->GetXaxis()->SetTitleFont(42);
  g->GetYaxis()->SetTitleFont(42);
  g->GetXaxis()->CenterTitle(true);
  g->GetYaxis()->CenterTitle(true);
  g->GetXaxis()->SetNdivisions(505);
}

/*! \fn beautify
 *  \brief utility function to beautify a TH1
 */

/*--------------------------------------------------------------------*/
void beautify(TH1 *h) {
  /*--------------------------------------------------------------------*/
  h->SetMinimum(0.);
  h->GetXaxis()->SetLabelFont(42);
  h->GetYaxis()->SetLabelFont(42);
  h->GetYaxis()->SetLabelSize(.055);
  h->GetXaxis()->SetLabelSize(.055);
  h->GetYaxis()->SetTitleSize(.055);
  h->GetXaxis()->SetTitleSize(.055);
  h->GetXaxis()->SetTitleOffset(1.1);
  h->GetYaxis()->SetTitleOffset(0.6);
  h->GetXaxis()->SetTitleFont(42);
  h->GetYaxis()->SetTitleFont(42);
  h->GetXaxis()->CenterTitle(true);
  h->GetYaxis()->CenterTitle(true);
  h->GetXaxis()->SetNdivisions(505);
}

/*! \fn adjustmargins
 *  \brief utility function to adjust margins of a TCanvas
 */

/*--------------------------------------------------------------------*/
void adjustmargins(TCanvas *canv) {
  /*--------------------------------------------------------------------*/
  canv->cd()->SetBottomMargin(0.14);
  canv->cd()->SetLeftMargin(0.07);
  canv->cd()->SetRightMargin(0.03);
  canv->cd()->SetTopMargin(0.06);
}

/*! \fn adjustmargins
 *  \brief utility function to adjust margins of a TVirtualPad
 */

/*--------------------------------------------------------------------*/
void adjustmargins(TVirtualPad *canv) {
  /*--------------------------------------------------------------------*/
  canv->SetBottomMargin(0.12);
  canv->SetLeftMargin(0.07);
  canv->SetRightMargin(0.01);
  canv->SetTopMargin(0.02);
}

/*! \fn checkTH1AndReturn
 *  \brief utility function to check if a TH1 exists in a TFile before returning it
 */

/*--------------------------------------------------------------------*/
TH1F *checkTH1AndReturn(TFile *f, TString address) {
  /*--------------------------------------------------------------------*/
  TH1F *h = NULL;
  if (f->GetListOfKeys()->Contains(address)) {
    h = (TH1F *)f->Get(address);
  }
  return h;
}

/*! \fn checkThView
 *  \brief utility function to return the pv::view corresponding to a given PV bias plot
 */

/*--------------------------------------------------------------------*/
pv::view checkTheView(const TString &toCheck) {
  /*--------------------------------------------------------------------*/
  if (toCheck.Contains("dxy")) {
    if (toCheck.Contains("phi") || toCheck.Contains("ladder")) {
      return pv::dxyphi;
    } else if (toCheck.Contains("eta") || toCheck.Contains("modZ")) {
      return pv::dxyeta;
    } else {
      return pv::pT;
    }
  } else if (toCheck.Contains("dz")) {
    if (toCheck.Contains("phi") || toCheck.Contains("ladder")) {
      return pv::dzphi;
    } else if (toCheck.Contains("eta") || toCheck.Contains("modZ")) {
      return pv::dzeta;
    } else {
      return pv::pT;
    }
  } else {
    return pv::generic;
  }
}

/*! \fn timify
 *  \brief utility function to make a histogram x-axis of the time type
 */

/*--------------------------------------------------------------------*/
template <typename T>
void timify(T *mgr)
/*--------------------------------------------------------------------*/
{
  mgr->GetXaxis()->SetTimeDisplay(1);
  mgr->GetXaxis()->SetNdivisions(510);
  mgr->GetXaxis()->SetTimeFormat("%Y-%m-%d");
  mgr->GetXaxis()->SetTimeOffset(0, "gmt");
  mgr->GetXaxis()->SetLabelSize(.035);
}

struct increase {
  template <class T>
  bool operator()(T const &a, T const &b) const {
    return a > b;
  }
};

/*! \fn getMaximumFromArray
 *  \brief utility function to extract the maximum out of an array of histograms
 */

/*--------------------------------------------------------------------*/
Double_t getMaximumFromArray(TObjArray *array)
/*--------------------------------------------------------------------*/
{
  Double_t theMaximum = -999.;  //(static_cast<TH1*>(array->At(0)))->GetMaximum();

  for (Int_t i = 0; i < array->GetSize(); i++) {
    double theMaxForThisHist;
    auto hist = static_cast<TH1 *>(array->At(i));
    std::vector<double> maxima;
    for (int j = 0; j < hist->GetNbinsX(); j++)
      maxima.push_back(hist->GetBinContent(j));
    std::sort(std::begin(maxima), std::end(maxima));  //,increase());
    double rms_maxima = TMath::RMS(hist->GetNbinsX(), &(maxima[0]));
    double mean_maxima = TMath::Mean(hist->GetNbinsX(), &(maxima[0]));

    const Int_t nq = 100;
    Double_t xq[nq];  // position where to compute the quantiles in [0,1]
    Double_t yq[nq];  // array to contain the quantiles
    for (Int_t i = 0; i < nq; i++)
      xq[i] = 0.9 + (Float_t(i + 1) / nq) * 0.10;
    TMath::Quantiles(maxima.size(), nq, &(maxima[0]), yq, xq);

    //for(int q=0;q<nq;q++){
    //  std::cout<<q<<" "<<xq[q]<<" "<<yq[q]<<std::endl;
    //}

    //for (const auto &element : maxima){
    //  if(element<1.5*mean_maxima){
    //	theMaxForThisHist=element;
    //	break;
    //  }
    //}

    theMaxForThisHist = yq[80];

    std::cout << "rms_maxima[" << i << "]" << rms_maxima << " mean maxima[" << mean_maxima
              << "] purged maximum:" << theMaxForThisHist << std::endl;

    if (theMaxForThisHist > theMaximum)
      theMaximum = theMaxForThisHist;

    /*
    if( (static_cast<TH1*>(array->At(i)))->GetMaximum() > theMaximum){
      theMaximum = (static_cast<TH1*>(array->At(i)))->GetMaximum();
      //cout<<"i= "<<i<<" theMaximum="<<theMaximum<<endl;
    }
    */
  }

  return theMaximum;
}

/*! \fn superImposeIOVBoundaries
 *  \brief utility function to superimpose the IOV boundaries on the trend plot
 */

/*--------------------------------------------------------------------*/
void superImposeIOVBoundaries(TCanvas *c,
                              bool lumi_axis_format,
                              bool time_axis_format,
                              const std::map<int, double> &lumiMapByRun,
                              const std::map<int, TDatime> &timeMap,
                              bool drawText)
/*--------------------------------------------------------------------*/
{
  // get the vector of runs in the lumiMap
  std::vector<int> vruns;
  for (auto const &imap : lumiMapByRun) {
    vruns.push_back(imap.first);
    //std::cout<<" run:" << imap.first << " lumi: "<< imap.second << std::endl;
  }

  //std::vector<vint> truns;
  for (auto const &imap : timeMap) {
    std::cout << " run:" << imap.first << " time: " << imap.second.Convert() << std::endl;
  }

  /* Run-2 Ultra-Legacy ReReco IOVs (from tag SiPixelTemplateDBObject_38T_v16_offline)
     1      271866   2016-04-28   2cc9ecb98e8ba900b26701d6adf052ba59c1f5e1  SiPixelTemplateDBObject 2016
     2      276315   2016-07-04   28a6077528012fe0abb03df9ef52e5976137c324  SiPixelTemplateDBObject
     3      278271   2016-08-05   43744865be66299a3a37599961a9faf5a0d2bd78  SiPixelTemplateDBObject
     4      280928   2016-09-16   7533fd5b60c10a9c55cddc6d2677d3fae6e8bb14  SiPixelTemplateDBObject
     5      290543   2017-03-30   05803622f45cf4ee2c07af2bb97875bd1010bfea  SiPixelTemplateDBObject 2017
     6      297281   2017-06-22   0ed971165dda7ae1db50c6636dab049087cad9a1  SiPixelTemplateDBObject
     7      298653   2017-07-09   e3af935c8f4eded04455a88959501d7408895501  SiPixelTemplateDBObject
     8      299443   2017-07-19   e2203ad6babf1885d754ae1392e4841a9b20c02e  SiPixelTemplateDBObject
     9      300389   2017-08-03   d512d75856a89b7602c143046f9e030e112c81e5  SiPixelTemplateDBObject
     10     301046   2017-08-12   c624d964356bf70df4a33761c32f8976a0d89257  SiPixelTemplateDBObject
     11     302131   2017-08-31   ca6f56bb82434b69bd951d5ca89cdce841473ce0  SiPixelTemplateDBObject
     12     303790   2017-09-23   cbb7b82c563a21f06bf7e57bec3a490625c98d18  SiPixelTemplateDBObject
     13     303998   2017-09-27   7aec56e15aed26f7b686535ecaff59911947f802  SiPixelTemplateDBObject
     14     304911   2017-10-13   d5bc4c312735d2d97e617ae3d45c735639a1cd82  SiPixelTemplateDBObject
     15     313041   2018-03-28   f3752b4a1fac4aa65c6439ab75d216dcc3ed27a9  SiPixelTemplateDBObject 2018
     16     314881   2018-04-22   bc8784a015d78068c51c9a581328b064299a31b4  SiPixelTemplateDBObject
     17     316758   2018-05-22   3a8abe693d1d3df829212e1049330e68f3392282  SiPixelTemplateDBObject
     18     317475   2018-06-05   83e09d34c122040bca0f5daf4b89a0435663c230  SiPixelTemplateDBObject
     19     317485   2018-06-05   3a8abe693d1d3df829212e1049330e68f3392282  SiPixelTemplateDBObject
     20     317527   2018-06-06   356b03fcd7e869ef8f28153318f0cd32c27b1f40  SiPixelTemplateDBObject
     21     317661   2018-06-10   1cbb2fc3edf448c50d00fac3aa47c8cf3b19ac6f  SiPixelTemplateDBObject
     22     317664   2018-06-11   356b03fcd7e869ef8f28153318f0cd32c27b1f40  SiPixelTemplateDBObject
     23     318227   2018-06-21   1cbb2fc3edf448c50d00fac3aa47c8cf3b19ac6f  SiPixelTemplateDBObject
     24     320377   2018-07-26   5e9cd265167ec543aac49685cf706a58014c08f8  SiPixelTemplateDBObject
     25     321831   2018-08-27   e47d453934b47f06e53a323bd9ae16b38ec1bbd1  SiPixelTemplateDBObject
     26     322510   2018-09-09   df783de5f30a99bc036d8973e48da78513206aba  SiPixelTemplateDBObject
     27     322603   2018-09-09   e47d453934b47f06e53a323bd9ae16b38ec1bbd1  SiPixelTemplateDBObject
     28     323232   2018-09-21   9e660fd65e392f01f69cde77a52f59e934d7737d  SiPixelTemplateDBObject
     29     324245   2018-10-08   6caa50b30aa80ede330585888cdc5e804853d1da  SiPixelTemplateDBObject
  */

  static const int nIOVs =
      29;  //   1       2       3       4       5       6       7       8       9      10      11      12      13      14      15
  int IOVboundaries[nIOVs] = {
      271866,
      276315,
      278271,
      280928,
      290543,
      297281,
      298653,
      299443,
      300389,
      301046,
      302131,
      303790,
      303998,
      304911,
      313041,
      //  16      17      18      19      20      21      22      23      24      25      26      27      28      29
      314881,
      316758,
      317475,
      317485,
      317527,
      317661,
      317664,
      318227,
      320377,
      321831,
      322510,
      322603,
      323232,
      324245};

  int benchmarkruns[nIOVs] = {271866, 276315, 278271, 280928, 290543, 297281, 298653, 299443, 300389, 301046,
                              302131, 303790, 303998, 304911, 313041, 314881, 316758, 317475, 317485, 317527,
                              317661, 317664, 318227, 320377, 321831, 322510, 322603, 323232, 324245};

  std::string dates[nIOVs] = {"2016-04-28", "2016-07-04", "2016-08-05", "2016-09-16", "2017-03-30", "2017-06-22",
                              "2017-07-09", "2017-07-19", "2017-08-03", "2017-08-12", "2017-08-31", "2017-09-23",
                              "2017-09-27", "2017-10-13", "2018-03-28", "2018-04-22", "2018-05-22", "2018-06-05",
                              "2018-06-05", "2018-06-06", "2018-06-10", "2018-06-11", "2018-06-21", "2018-07-26",
                              "2018-08-27", "2018-09-09", "2018-09-09", "2018-09-21", "2018-10-08"};

  TArrow *IOV_lines[nIOVs];
  c->cd();
  c->Update();

  TArrow *a_lines[nIOVs];
  TArrow *b_lines[nIOVs];
  for (Int_t IOV = 0; IOV < nIOVs; IOV++) {
    // check we are not in the RMS histogram to avoid first line
    if (IOVboundaries[IOV] < vruns.front())
      continue;
    //&& ((TString)c->GetName()).Contains("RMS")) continue;
    int closestrun = pv::closest(vruns, IOVboundaries[IOV]);
    int closestbenchmark = pv::closest(vruns, benchmarkruns[IOV]);

    if (lumi_axis_format) {
      if (closestrun < 0)
        continue;
      //std::cout<< "natural boundary: " << IOVboundaries[IOV] << " closest:" << closestrun << std::endl;

      a_lines[IOV] = new TArrow(
          lumiMapByRun.at(closestrun), (c->GetUymin()), lumiMapByRun.at(closestrun), c->GetUymax(), 0.5, "|>");

      if (closestbenchmark < 0)
        continue;
      b_lines[IOV] = new TArrow(lumiMapByRun.at(closestbenchmark),
                                (c->GetUymin()),
                                lumiMapByRun.at(closestbenchmark),
                                c->GetUymax(),
                                0.5,
                                "|>");

    } else if (time_axis_format) {
      if (closestrun < 0)
        continue;
      std::cout << "natural boundary: " << IOVboundaries[IOV] << " closest:" << closestrun << std::endl;
      a_lines[IOV] = new TArrow(
          timeMap.at(closestrun).Convert(), (c->GetUymin()), timeMap.at(closestrun).Convert(), c->GetUymax(), 0.5, "|>");

      if (closestbenchmark < 0)
        continue;
      b_lines[IOV] = new TArrow(timeMap.at(closestbenchmark).Convert(),
                                (c->GetUymin()),
                                timeMap.at(closestbenchmark).Convert(),
                                c->GetUymax(),
                                0.5,
                                "|>");

    } else {
      a_lines[IOV] = new TArrow(IOVboundaries[IOV],
                                (c->GetUymin()),
                                IOVboundaries[IOV],
                                0.65 * c->GetUymax(),
                                0.5,
                                "|>");  //(c->GetUymin()+0.2*(c->GetUymax()-c->GetUymin()) ),0.5,"|>");
      b_lines[IOV] = new TArrow(benchmarkruns[IOV],
                                (c->GetUymin()),
                                benchmarkruns[IOV],
                                0.65 * c->GetUymax(),
                                0.5,
                                "|>");  //(c->GetUymin()+0.2*(c->GetUymax()-c->GetUymin()) ),0.5,"|>");
    }
    a_lines[IOV]->SetLineColor(kBlue);
    a_lines[IOV]->SetLineStyle(9);
    a_lines[IOV]->SetLineWidth(1);
    a_lines[IOV]->Draw("same");

    b_lines[IOV]->SetLineColor(kGray);
    b_lines[IOV]->SetLineStyle(1);
    b_lines[IOV]->SetLineWidth(2);
    //b_lines[IOV]->Draw("same");
  }

  TPaveText *runnumbers[nIOVs];

  for (Int_t IOV = 0; IOV < nIOVs; IOV++) {
    if (IOVboundaries[IOV] < vruns.front())
      continue;
    //&& ((TString)c->GetName()).Contains("RMS")) continue;
    int closestrun = pv::closest(vruns, IOVboundaries[IOV]);

    Int_t ix1;
    Int_t ix2;
    Int_t iw = gPad->GetWw();
    Int_t ih = gPad->GetWh();
    Double_t x1p, y1p, x2p, y2p;
    gPad->GetPadPar(x1p, y1p, x2p, y2p);
    ix1 = (Int_t)(iw * x1p);
    ix2 = (Int_t)(iw * x2p);
    Double_t wndc = TMath::Min(1., (Double_t)iw / (Double_t)ih);
    Double_t rw = wndc / (Double_t)iw;
    Double_t x1ndc = (Double_t)ix1 * rw;
    Double_t x2ndc = (Double_t)ix2 * rw;
    Double_t rx1, ry1, rx2, ry2;
    gPad->GetRange(rx1, ry1, rx2, ry2);
    Double_t rx = (x2ndc - x1ndc) / (rx2 - rx1);
    Double_t _sx;
    if (lumi_axis_format) {
      if (closestrun < 0)
        break;
      _sx = rx * (lumiMapByRun.at(closestrun) - rx1) + x1ndc;  //-0.05;
    } else if (time_axis_format) {
      if (closestrun < 0)
        break;
      _sx = rx * (timeMap.at(closestrun).Convert() - rx1) + x1ndc;  //-0.05;
    } else {
      _sx = rx * (IOVboundaries[IOV] - rx1) + x1ndc;  //-0.05
    }
    Double_t _dx = _sx + 0.05;

    Int_t index = IOV % 5;
    // if(IOV<5)
    //   index=IOV;
    // else{
    //   index=IOV-5;
    // }

    //runnumbers[IOV] = new TPaveText(_sx+0.001,0.14+(0.03*index),_dx,(0.17+0.03*index),"blNDC");
    runnumbers[IOV] = new TPaveText(_sx + 0.001, 0.89 - (0.05 * index), _dx + 0.024, (0.92 - 0.05 * index), "blNDC");
    //runnumbers[IOV]->SetTextAlign(11);
    TText *textRun = runnumbers[IOV]->AddText(Form("%i", int(IOVboundaries[IOV])));
    //TText *textRun = runnumbers[IOV]->AddText(Form("%s",dates[IOV].c_str()));
    textRun->SetTextSize(0.038);
    textRun->SetTextColor(kBlue);
    runnumbers[IOV]->SetFillColor(10);
    runnumbers[IOV]->SetLineColor(kBlue);
    runnumbers[IOV]->SetBorderSize(1);
    runnumbers[IOV]->SetLineWidth(1);
    runnumbers[IOV]->SetTextColor(kBlue);
    runnumbers[IOV]->SetTextFont(42);
    if (drawText)
      runnumbers[IOV]->Draw("same");
  }
}

/*! \fn processData
 *  \brief function where the magic happens, take the raw inputs and creates the output trends
 */

/*--------------------------------------------------------------------*/
outTrends processData(size_t iter,
                      std::vector<int> intersection,
                      const Int_t nDirs_,
                      const char *dirs[10],
                      TString LegLabels[10],
                      bool useRMS)
/*--------------------------------------------------------------------*/
{
  outTrends ret;
  ret.init();

  unsigned int effSize = std::min(nWorkers, intersection.size());

  unsigned int pitch = std::floor(intersection.size() / effSize);
  unsigned int first = iter * pitch;
  unsigned int last = (iter == (effSize - 1)) ? intersection.size() : ((iter + 1) * pitch);

  std::cout << "iter:" << iter << "| pitch: " << pitch << " [" << first << "-" << last << ")" << std::endl;

  ret.m_index = iter;

  for (unsigned int n = first; n < last; n++) {
    //in case of debug, use only 50
    //for(unsigned int n=0; n<50;n++){

    //if(intersection.at(n)!=283946)
    //  continue;

    std::cout << "iter: " << iter << " " << n << " " << intersection.at(n) << std::endl;

    TFile *fins[nDirs_];

    TH1F *dxyPhiMeanTrend[nDirs_];
    TH1F *dxyPhiWidthTrend[nDirs_];
    TH1F *dzPhiMeanTrend[nDirs_];
    TH1F *dzPhiWidthTrend[nDirs_];

    TH1F *dxyLadderMeanTrend[nDirs_];
    TH1F *dxyLadderWidthTrend[nDirs_];
    TH1F *dzLadderWidthTrend[nDirs_];
    TH1F *dzLadderMeanTrend[nDirs_];

    TH1F *dxyModZMeanTrend[nDirs_];
    TH1F *dxyModZWidthTrend[nDirs_];
    TH1F *dzModZMeanTrend[nDirs_];
    TH1F *dzModZWidthTrend[nDirs_];

    TH1F *dxyEtaMeanTrend[nDirs_];
    TH1F *dxyEtaWidthTrend[nDirs_];
    TH1F *dzEtaMeanTrend[nDirs_];
    TH1F *dzEtaWidthTrend[nDirs_];

    TH1F *dxyNormPhiWidthTrend[nDirs_];
    TH1F *dxyNormEtaWidthTrend[nDirs_];
    TH1F *dzNormPhiWidthTrend[nDirs_];
    TH1F *dzNormEtaWidthTrend[nDirs_];

    TH1F *dxyNormPtWidthTrend[nDirs_];
    TH1F *dzNormPtWidthTrend[nDirs_];
    TH1F *dxyPtWidthTrend[nDirs_];
    TH1F *dzPtWidthTrend[nDirs_];

    TH1F *dxyIntegralTrend[nDirs_];
    TH1F *dzIntegralTrend[nDirs_];

    bool areAllFilesOK = true;
    Int_t lastOpen = 0;

    // loop over the objects
    for (Int_t j = 0; j < nDirs_; j++) {
      //fins[j] = TFile::Open(Form("%s/PVValidation_%s_%i.root",dirs[j],dirs[j],intersection[n]));

      size_t position = std::string(dirs[j]).find("/");
      string stem = std::string(dirs[j]).substr(position + 1);  // get from position to the end

      fins[j] = new TFile(Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n]));
      if (fins[j]->IsZombie()) {
        std::cout << Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n])
                  << " is a Zombie! cannot combine" << std::endl;
        areAllFilesOK = false;
        lastOpen = j;
        break;
      }

      std::cout << Form("%s/PVValidation_%s_%i.root", dirs[j], stem.c_str(), intersection[n])
                << " has size: " << fins[j]->GetSize() << " b ";

      // sanity check
      TH1F *h_tracks = (TH1F *)fins[j]->Get("PVValidation/EventFeatures/h_nTracks");
      if (j == 0) {
        TH1F *h_lumi = (TH1F *)fins[j]->Get("PVValidation/EventFeatures/h_lumiFromConfig");
        double lumi = h_lumi->GetBinContent(1);
        ret.m_lumiSoFar += lumi;
        //std::cout<<"lumi: "<<lumi
        //	 <<" ,lumi so far: "<<lumiSoFar<<std::endl;

        //outfile<<"run "<<intersection[n]<<" lumi: "<<lumi
        //     <<" ,lumi so far: "<<lumiSoFar<<std::endl;
      }

      Double_t numEvents = h_tracks->GetEntries();
      if (numEvents < 2500) {
        std::cout << "excluding " << intersection[n] << " because it has less than 2.5k events" << std::endl;
        areAllFilesOK = false;
        lastOpen = j;
        break;
      }

      dxyPhiMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_phi");
      //dxyPhiMeanTrend[j]     = checkTH1AndReturn(fins[j],"PVValidation/MeanTrends/means_dxy_phi");
      dxyPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_phi");
      dzPhiMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_phi");
      dzPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_phi");

      dxyLadderMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_ladder");
      dxyLadderWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_ladder");
      dzLadderMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_ladder");
      dzLadderWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_ladder");

      dxyEtaMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_eta");
      dxyEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_eta");
      dzEtaMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_eta");
      dzEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_eta");

      dxyModZMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dxy_modZ");
      dxyModZWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_modZ");
      dzModZMeanTrend[j] = (TH1F *)fins[j]->Get("PVValidation/MeanTrends/means_dz_modZ");
      dzModZWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_modZ");

      dxyNormPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_phi");
      dxyNormEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_eta");
      dzNormPhiWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_phi");
      dzNormEtaWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_eta");

      dxyNormPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dxy_pTCentral");
      dzNormPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/norm_widths_dz_pTCentral");
      dxyPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dxy_pTCentral");
      dzPtWidthTrend[j] = (TH1F *)fins[j]->Get("PVValidation/WidthTrends/widths_dz_pTCentral");

      dxyIntegralTrend[j] = (TH1F *)fins[j]->Get("PVValidation/ProbeTrackFeatures/h_probedxyRefitV");
      dzIntegralTrend[j] = (TH1F *)fins[j]->Get("PVValidation/ProbeTrackFeatures/h_probedzRefitV");

      // fill the vectors of biases

      auto dxyPhiBiases = getBiases(dxyPhiMeanTrend[j]);

      //std::cout<<"\n" <<j<<" "<< LegLabels[j] << " dxy(phi) mean: "<< dxyPhiBiases.getWeightedMean()
      //       <<" dxy(phi) max: "<< dxyPhiBiases.getMax()
      //       <<" dxy(phi) min: "<< dxyPhiBiases.getMin()
      //       << std::endl;

      ret.m_dxyPhiMeans[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean());
      ret.m_dxyPhiChi2[LegLabels[j]].push_back(TMath::Log10(dxyPhiBiases.getNormChi2()));
      ret.m_dxyPhiKS[LegLabels[j]].push_back(dxyPhiBiases.getKSScore());

      //std::cout<<"\n" <<j<<" "<< LegLabels[j] << " dxy(phi) ks score: "<< dxyPhiBiases.getKSScore() << std::endl;

      useRMS
          ? ret.m_dxyPhiLo[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean() - 2 * dxyPhiBiases.getWeightedRMS())
          : ret.m_dxyPhiLo[LegLabels[j]].push_back(dxyPhiBiases.getMin());
      useRMS
          ? ret.m_dxyPhiHi[LegLabels[j]].push_back(dxyPhiBiases.getWeightedMean() + 2 * dxyPhiBiases.getWeightedRMS())
          : ret.m_dxyPhiHi[LegLabels[j]].push_back(dxyPhiBiases.getMax());

      auto dxyEtaBiases = getBiases(dxyEtaMeanTrend[j]);
      ret.m_dxyEtaMeans[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean());
      ret.m_dxyEtaChi2[LegLabels[j]].push_back(TMath::Log10(dxyEtaBiases.getNormChi2()));
      ret.m_dxyEtaKS[LegLabels[j]].push_back(dxyEtaBiases.getKSScore());
      useRMS
          ? ret.m_dxyEtaLo[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean() - 2 * dxyEtaBiases.getWeightedRMS())
          : ret.m_dxyEtaLo[LegLabels[j]].push_back(dxyEtaBiases.getMin());
      useRMS
          ? ret.m_dxyEtaHi[LegLabels[j]].push_back(dxyEtaBiases.getWeightedMean() + 2 * dxyEtaBiases.getWeightedRMS())
          : ret.m_dxyEtaHi[LegLabels[j]].push_back(dxyEtaBiases.getMax());

      auto dzPhiBiases = getBiases(dzPhiMeanTrend[j]);
      ret.m_dzPhiMeans[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean());
      ret.m_dzPhiChi2[LegLabels[j]].push_back(TMath::Log10(dzPhiBiases.getNormChi2()));
      ret.m_dzPhiKS[LegLabels[j]].push_back(dzPhiBiases.getKSScore());
      useRMS ? ret.m_dzPhiLo[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean() - 2 * dzPhiBiases.getWeightedRMS())
             : ret.m_dzPhiLo[LegLabels[j]].push_back(dzPhiBiases.getMin());
      useRMS ? ret.m_dzPhiHi[LegLabels[j]].push_back(dzPhiBiases.getWeightedMean() + 2 * dzPhiBiases.getWeightedRMS())
             : ret.m_dzPhiHi[LegLabels[j]].push_back(dzPhiBiases.getMax());

      auto dzEtaBiases = getBiases(dzEtaMeanTrend[j]);
      ret.m_dzEtaMeans[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean());
      ret.m_dzEtaChi2[LegLabels[j]].push_back(TMath::Log10(dzEtaBiases.getNormChi2()));
      ret.m_dzEtaKS[LegLabels[j]].push_back(dzEtaBiases.getKSScore());
      useRMS ? ret.m_dzEtaLo[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean() - 2 * dzEtaBiases.getWeightedRMS())
             : ret.m_dzEtaLo[LegLabels[j]].push_back(dzEtaBiases.getMin());
      useRMS ? ret.m_dzEtaHi[LegLabels[j]].push_back(dzEtaBiases.getWeightedMean() + 2 * dzEtaBiases.getWeightedRMS())
             : ret.m_dzEtaHi[LegLabels[j]].push_back(dzEtaBiases.getMax());

      // unrolled histograms
      ret.m_dxyVect[LegLabels[j]].push_back(getUnrolledHisto(dxyIntegralTrend[j]));
      ret.m_dzVect[LegLabels[j]].push_back(getUnrolledHisto(dzIntegralTrend[j]));

      //std::cout<<std::endl;
      //std::cout<<" n. bins: "<< dxyVect[LegLabels[j]].back().get_n_bins()
      //       <<" y-min:   "<< dxyVect[LegLabels[j]].back().get_y_min()
      //       <<" y-max:   "<< dxyVect[LegLabels[j]].back().get_y_max() << std::endl;

      // beautify the histograms
      MakeNiceTrendPlotStyle(dxyPhiMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dxyPhiWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzPhiMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dzPhiWidthTrend[j], j);

      MakeNiceTrendPlotStyle(dxyLadderMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dxyLadderWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzLadderMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dzLadderWidthTrend[j], j);

      MakeNiceTrendPlotStyle(dxyEtaMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dxyEtaWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzEtaMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dzEtaWidthTrend[j], j);

      MakeNiceTrendPlotStyle(dxyModZMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dxyModZWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzModZMeanTrend[j], j);
      MakeNiceTrendPlotStyle(dzModZWidthTrend[j], j);

      MakeNiceTrendPlotStyle(dxyNormPhiWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dxyNormEtaWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzNormPhiWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzNormEtaWidthTrend[j], j);

      MakeNiceTrendPlotStyle(dxyNormPtWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzNormPtWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dxyPtWidthTrend[j], j);
      MakeNiceTrendPlotStyle(dzPtWidthTrend[j], j);
    }

    if (!areAllFilesOK) {
      // do all the necessary deletions
      std::cout << "\n====> not all files are OK" << std::endl;

      for (int i = 0; i < lastOpen; i++) {
        fins[i]->Close();
      }
      continue;
    } else {
      ret.m_runs.push_back(intersection.at(n));
      // push back the vector of lumi (in fb at this point)
      ret.m_lumiByRun.push_back(ret.m_lumiSoFar / 1000.);
      ret.m_lumiMapByRun[intersection.at(n)] = ret.m_lumiSoFar / 1000.;
    }

    std::cout << "I am still here - runs.size(): " << ret.m_runs.size() << std::endl;

    // Bias plots

    TCanvas *BiasesCanvas = new TCanvas(Form("Biases_%i", intersection.at(n)), "Biases", 1200, 1200);
    arrangeOutCanvas(BiasesCanvas,
                     dxyPhiMeanTrend,
                     dzPhiMeanTrend,
                     dxyEtaMeanTrend,
                     dzEtaMeanTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    BiasesCanvas->SaveAs(Form("Biases_%i.pdf", intersection.at(n)));
    BiasesCanvas->SaveAs(Form("Biases_%i.png", intersection.at(n)));

    // Bias vs L1 modules position

    TCanvas *BiasesL1Canvas = new TCanvas(Form("BiasesL1_%i", intersection.at(n)), "BiasesL1", 1200, 1200);
    arrangeOutCanvas(BiasesL1Canvas,
                     dxyLadderMeanTrend,
                     dzLadderMeanTrend,
                     dxyModZMeanTrend,
                     dzModZMeanTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    BiasesL1Canvas->SaveAs(Form("BiasesL1_%i.pdf", intersection.at(n)));
    BiasesL1Canvas->SaveAs(Form("BiasesL1_%i.png", intersection.at(n)));

    // Resolution plots

    TCanvas *ResolutionsCanvas = new TCanvas(Form("Resolutions_%i", intersection.at(n)), "Resolutions", 1200, 1200);
    arrangeOutCanvas(ResolutionsCanvas,
                     dxyPhiWidthTrend,
                     dzPhiWidthTrend,
                     dxyEtaWidthTrend,
                     dzEtaWidthTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    ResolutionsCanvas->SaveAs(Form("Resolutions_%i.pdf", intersection.at(n)));
    ResolutionsCanvas->SaveAs(Form("Resolutions_%i.png", intersection.at(n)));

    // Resolution plots vs L1 modules position

    TCanvas *ResolutionsL1Canvas = new TCanvas(Form("ResolutionsL1_%i", intersection.at(n)), "Resolutions", 1200, 1200);
    arrangeOutCanvas(ResolutionsL1Canvas,
                     dxyLadderWidthTrend,
                     dzLadderWidthTrend,
                     dxyModZWidthTrend,
                     dzModZWidthTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    ResolutionsL1Canvas->SaveAs(Form("ResolutionsL1_%i.pdf", intersection.at(n)));
    ResolutionsL1Canvas->SaveAs(Form("ResolutionsL1_%i.png", intersection.at(n)));

    // Pull plots

    TCanvas *PullsCanvas = new TCanvas(Form("Pulls_%i", intersection.at(n)), "Pulls", 1200, 1200);
    arrangeOutCanvas(PullsCanvas,
                     dxyNormPhiWidthTrend,
                     dzNormPhiWidthTrend,
                     dxyNormEtaWidthTrend,
                     dzNormEtaWidthTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    PullsCanvas->SaveAs(Form("Pulls_%i.pdf", intersection.at(n)));
    PullsCanvas->SaveAs(Form("Pulls_%i.png", intersection.at(n)));

    // pT plots

    TCanvas *ResolutionsVsPt =
        new TCanvas(Form("ResolutionsVsPT_%i", intersection.at(n)), "ResolutionsVsPt", 1200, 1200);
    arrangeOutCanvas(ResolutionsVsPt,
                     dxyPtWidthTrend,
                     dzPtWidthTrend,
                     dxyNormPtWidthTrend,
                     dzNormPtWidthTrend,
                     nDirs_,
                     LegLabels,
                     intersection.at(n));

    ResolutionsVsPt->SaveAs(Form("ResolutionsVsPt_%i.pdf", intersection.at(n)));
    ResolutionsVsPt->SaveAs(Form("ResolutionsVsPt_%i.png", intersection.at(n)));

    // do all the necessary deletions

    for (int i = 0; i < nDirs_; i++) {
      delete dxyPhiMeanTrend[i];
      delete dzPhiMeanTrend[i];
      delete dxyEtaMeanTrend[i];
      delete dzEtaMeanTrend[i];

      delete dxyPhiWidthTrend[i];
      delete dzPhiWidthTrend[i];
      delete dxyEtaWidthTrend[i];
      delete dzEtaWidthTrend[i];

      delete dxyNormPhiWidthTrend[i];
      delete dxyNormEtaWidthTrend[i];
      delete dzNormPhiWidthTrend[i];
      delete dzNormEtaWidthTrend[i];

      delete dxyNormPtWidthTrend[i];
      delete dzNormPtWidthTrend[i];
      delete dxyPtWidthTrend[i];
      delete dzPtWidthTrend[i];

      fins[i]->Close();
    }

    delete BiasesCanvas;
    delete BiasesL1Canvas;
    delete ResolutionsCanvas;
    delete ResolutionsL1Canvas;
    delete PullsCanvas;
    delete ResolutionsVsPt;

    std::cout << std::endl;
  }

  return ret;
}
