#ifndef ALIGNMENT_OFFLINEVALIDATION_PREPAREPVTRENDS_H_
#define ALIGNMENT_OFFLINEVALIDATION_PREPAREPVTRENDS_H_

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
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

#include "Alignment/OfflineValidation/interface/OutPVtrends.h"

/*!
 * \def some convenient I/O
 */
#define logInfo std::cout << "INFO: "
#define logWarning std::cout << "WARNING: "
#define logError std::cout << "ERROR!!! "

/*!
 * \def boolean to decide if it is in debug mode
 */
#define VERBOSE false

namespace pv {
  enum view { dxyphi, dzphi, dxyeta, dzeta, pT, generic };

  /*! \fn closest
   *  \brief method to find first value that doesn not compare left
   */

  inline int closest(std::vector<int> const &vec, int value) {
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
      logInfo << "pv::wrappedTrends c'tor" << std::endl;

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
   */

  struct bundle {
    bundle(int nObjects, const TString &dataType, const TString &dataTypeLabel, const bool &useRMS) {
      m_nObjects = nObjects;
      m_datatype = dataType.Data();
      m_datatypelabel = dataTypeLabel.Data();
      m_useRMS = useRMS;

      logInfo << "pv::bundle c'tor: " << dataTypeLabel << " member: " << m_datatypelabel << std::endl;

      logInfo << m_axis_types << std::endl;
    }

    int getNObjects() const { return m_nObjects; }
    const char *getDataType() const { return m_datatype; }
    const char *getDataTypeLabel() const { return m_datatypelabel; }
    bool isUsingRMS() const { return m_useRMS; }
    void printAll() {
      logInfo << "dataType      " << m_datatype << std::endl;
      logInfo << "dataTypeLabel " << m_datatypelabel << std::endl;
    }

  private:
    int m_nObjects;
    const char *m_datatype;
    const char *m_datatypelabel;
    std::bitset<2> m_axis_types;
    bool m_useRMS;
  };

}  // namespace pv

class PreparePVTrends {
public:
  PreparePVTrends(const char *outputFileName, int nWorkers, boost::property_tree::ptree &json);
  ~PreparePVTrends() {}

  void setDirsAndLabels(boost::property_tree::ptree &json);

  void multiRunPVValidation(bool useRMS = true, TString lumiInputFile = "", bool doUnitTest = false);

  static pv::biases getBiases(TH1F *hist);
  static unrolledHisto getUnrolledHisto(TH1F *hist);
  static TH1F *drawConstantWithErr(TH1F *hist, Int_t iter, Double_t theConst);
  static outPVtrends processData(size_t iter,
                                 std::vector<int> intersection,
                                 const Int_t nDirs_,
                                 const char *dirs[10],
                                 TString LegLabels[10],
                                 bool useRMS,
                                 const size_t nWorkers,
                                 bool doUnitTest);
  std::vector<int> list_files(const char *dirname = ".", const char *ext = ".root");
  void outputGraphs(const pv::wrappedTrends &allInputs,
                    const std::vector<double> &ticks,
                    const std::vector<double> &ex_ticks,
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
                    const TString &label);

private:
  const char *outputFileName_;
  const size_t nWorkers_;  //def number of workers
  std::vector<std::string> DirList;
  std::vector<std::string> LabelList;
};

#endif  // ALIGNMENT_OFFLINEVALIDATION_PREPAREPVTRENDS_H_
