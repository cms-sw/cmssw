#ifndef PREPAREDMRTRENDS_H_
#define PREPAREDMRTRENDS_H_

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <iomanip>
#include <fstream>
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
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TMath.h"
#include "TRegexp.h"
#include "TPaveLabel.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TLine.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

/*!
 * \def Dummy value in case a DMR would fail for instance
 */
#define DUMMY -999.
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

/*! \class Geometry
 *  \brief Class Geometry
 *         Contains vector for fit parameters (mean, sigma, etc.) obtained from multiple IOVs
 *         See Structure Point for description of the parameters.
 */

class Geometry {
public:
  std::vector<Point> points;

private:
  //template<typename T> std::vector<T> GetQuantity (T (Point::*getter)() const) const {
  std::vector<float> GetQuantity(float (Point::*getter)() const) const {
    std::vector<float> v;
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
  std::vector<float> Run() const { return GetQuantity(&Point::GetRun); }
  std::vector<float> Mu() const { return GetQuantity(&Point::GetMu); }
  std::vector<float> MuPlus() const { return GetQuantity(&Point::GetMuPlus); }
  std::vector<float> MuMinus() const { return GetQuantity(&Point::GetMuMinus); }
  std::vector<float> Sigma() const { return GetQuantity(&Point::GetSigma); }
  std::vector<float> SigmaPlus() const { return GetQuantity(&Point::GetSigmaPlus); }
  std::vector<float> SigmaMinus() const { return GetQuantity(&Point::GetSigmaMinus); }
  std::vector<float> DeltaMu() const { return GetQuantity(&Point::GetDeltaMu); }
  std::vector<float> SigmaDeltaMu() const { return GetQuantity(&Point::GetSigmaDeltaMu); }
  //std::vector<float> Graph (std::string variable) const {
  // };
};

class PrepareDMRTrends {
 public:

  PrepareDMRTrends(TString outputdir, boost::property_tree::ptree& json);

  TString getName(TString structure, int layer, TString geometry);
  const std::map<TString, int> numberOfLayers(TString Year);
  void compileDMRTrends(std::vector<int> IOVlist,
			TString Variable,
			std::vector<std::string> labels,
			TString Year,
			std::vector<std::string> inputFiles,
			bool FORCE = false);

 private:
  TString outputdir_;
  std::vector<std::string> geometries;

};

#endif // PREPAREDMRTRENDS_H_
