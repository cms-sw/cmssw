#ifndef ALIGNMENT_OFFLINEVALIDATION_PVVALIDATIONHELPER_H
#define ALIGNMENT_OFFLINEVALIDATION_PVVALIDATIONHELPER_H

#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <utility>
#include "TH1.h"

#ifndef PLOTTING_MACRO
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define COUT edm::LogWarning("PVValidationHelpers")
#else
#include <iostream>
#define COUT std::cout << "PVValidationHelpers: "
#endif

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

namespace PVValHelper {

  const double max_eta_phase0 = 2.5;
  const double max_eta_phase1 = 2.7;

  // helper logarithmic bin generator

  template <typename T, size_t N>
  std::array<T, N + 1> makeLogBins(const T& min, const T& max) {
    const T minLog10 = std::log10(min);
    const T maxLog10 = std::log10(max);
    const T width = (maxLog10 - minLog10) / N;
    std::array<T, N + 1> ret;
    ret[0] = std::pow(10, minLog10);
    const T mult = std::pow(10, width);
    for (size_t i = 1; i <= N; ++i) {
      ret[i] = ret[i - 1] * mult;
    }
    return ret;
  }

  // helper data formats

  enum estimator { MEAN = 1, WIDTH = 2, MEDIAN = 3, MAD = 4, UNKWN = -1 };

  enum residualType {
    dxy = 1,
    dx = 2,
    dy = 3,
    dz = 4,
    IP2D = 5,
    resz = 6,
    IP3D = 7,
    d3D = 8,

    norm_dxy = 9,
    norm_dx = 10,
    norm_dy = 11,
    norm_dz = 12,
    norm_IP2D = 13,
    norm_resz = 14,
    norm_IP3D = 15,
    norm_d3D = 16,
    END_OF_TYPES = 17,
  };

  enum plotVariable {
    phi = 1,
    eta = 2,
    pT = 3,
    pTCentral = 4,
    ladder = 5,
    modZ = 6,
    END_OF_PLOTS = 7,
  };

  struct histodetails {
    int histobins;
    std::map<std::pair<residualType, plotVariable>, std::pair<float, float>> range;
    std::map<plotVariable, std::vector<float>> trendbins;

    void setMap(residualType type, plotVariable plot, float low, float high) {
      assert(low < high);
      range[std::make_pair(type, plot)] = std::make_pair(low, high);
    }

    std::pair<float, float> getRange(residualType type, plotVariable plot) {
      if (range.find(std::make_pair(type, plot)) != range.end()) {
        return range[std::make_pair(type, plot)];
      } else {
        COUT << "Trying to get range for non-existent combination " << std::endl;
        return std::make_pair(0., 0.);
      }
    }

    float getLow(residualType type, plotVariable plot) {
      if (range.find(std::make_pair(type, plot)) != range.end()) {
        return range[std::make_pair(type, plot)].first;
      } else {
        COUT << "Trying to get low end of range for non-existent combination " << std::endl;
        return 0.;
      }
    }

    float getHigh(residualType type, plotVariable plot) {
      if (range.find(std::make_pair(type, plot)) != range.end()) {
        return range[std::make_pair(type, plot)].second;
      } else {
        COUT << "Trying get high end of range for non-existent combination " << std::endl;
        return 0.;
      }
    }
  };

  typedef std::tuple<std::string, std::string, std::string> plotLabels;

  // helper methods

  void add(std::map<std::string, TH1*>& h, TH1* hist);

  void fill(std::map<std::string, TH1*>& h, const std::string& s, double x);

  void fill(std::map<std::string, TH1*>& h, const std::string& s, double x, double y);

  void fillByIndex(std::vector<TH1F*>& h, unsigned int index, double x, std::string tag = "");

  void shrinkHistVectorToFit(std::vector<TH1F*>& h, unsigned int desired_size);

  plotLabels getTypeString(residualType type);

  plotLabels getVarString(plotVariable var);

  std::vector<float> generateBins(int n, float start, float range);

  Measurement1D getMedian(TH1F* histo);

  Measurement1D getMAD(TH1F* histo);

  std::pair<Measurement1D, Measurement1D> fitResiduals(TH1* hist);

};  // namespace PVValHelper

#endif
