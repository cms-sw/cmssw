#ifndef CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h
#define CalibTracker_SiStripLorentzAngle_SiStripLorentzAngleCalibrationHelper_h

// user includes
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system includes
#include <string>

// for ROOT
#include "TString.h"
#include "TFitResult.h"
#include "TF1.h"

namespace siStripLACalibration {

  /**
   * @brief Generates a module location type string based on the detector ID and topology information.
   *
   * Given a module ID and the corresponding TrackerTopology, this function constructs a module
   * location type string in the format "subdet_LlayerType", where subdet is the subdetector name (TIB or TOB),
   * layer is the layer number, and Type is 'a' for axial or 's' for stereo.
   *
   * @param mod The module ID.
   * @param tTopo Pointer to the TrackerTopology object providing information about the detector.
   * @return A module location type string.
   */

  //_____________________________________________________________________
  inline std::string moduleLocationType(const uint32_t& mod, const TrackerTopology* tTopo) {
    const SiStripDetId detid(mod);
    std::string subdet = "";
    unsigned int layer = 0;
    if (detid.subDetector() == SiStripDetId::TIB) {
      subdet = "TIB";
      layer = tTopo->layer(mod);
    } else if (detid.subDetector() == SiStripDetId::TOB) {
      subdet = "TOB";
      layer = tTopo->layer(mod);
    }

    if (layer == 0)
      return subdet;

    std::string type = (detid.stereo() ? "s" : "a");
    std::string d_l_t = Form("%s_L%d%s", subdet.c_str(), layer, type.c_str());
    return d_l_t;
  }

  /**
   * @brief Process a string in the format "subdet_LlayerType" and compute values.
   *
   * This function takes a string in the format "subdet_LlayerType" and parses it to extract
   * information about the layer and type. It then computes and returns a std::pair<int, int> where
   * the first element is 1 if type is "a" and 2 if type is "s", 
   * and the second element is the processed value of layer if subdet is "TIB" or layer + 4 if subdet is "TOB".
   *
   * @param locType The input string in the format "subdet_LlayerType".
   * @return A std::pair<int, int> containing the processed values. If the input format is invalid,
   *         the pair (-1, -1) is returned.
   *
   * @example
   *   std::string d_l_t = "TIB_L3a";
   *   std::pair<int, int> result = processString(d_l_t);
   *   // The result will contain processed values based on the input.
   */

  //_____________________________________________________________________
  inline std::pair<int, int> locationTypeIndex(const std::string& locType) {
    // Assuming input format is "subdet_LlayerType"
    // Example: "TIB_L3a"

    std::string subdet, layerType;
    int layer;

    // Parse the input string
    if (sscanf(locType.c_str(), "%3s_L%d%1[a-zA-Z]", &subdet[0], &layer, &layerType[0]) == 3) {
      // Process subdet and layerType to compute the values
      LogTrace("locationTypeIndex") << "subdet " << &subdet[0] << ") layer " << layer << " type " << layerType[0]
                                    << std::endl;

      int firstElement = (layerType[0] == 'a') ? 1 : 2;
      int secondElement = (std::string(&subdet[0]) == "TIB") ? layer : (layer + 4);

      return std::make_pair(firstElement, secondElement);
    } else {
      // Handle invalid input format
      // FIXME use MessageLogger
      std::cerr << "Invalid input format: " << locType << std::endl;
      return std::make_pair(-1, -1);  // Indicates error
    }
  }

  // SiStripLatency::singleReadOutMode() returns
  // 1: all in peak, 0: all in deco, -1: mixed state
  enum { k_DeconvolutionMode = 0, k_PeakMode = 1 };

  //_____________________________________________________________________
  inline const std::string fieldAsString(const float& inputField) {
    std::string theMagFieldStr = std::to_string(inputField);
    size_t dotPosition = theMagFieldStr.find('.');
    if (dotPosition != std::string::npos) {
      theMagFieldStr = theMagFieldStr.substr(0, dotPosition + 2);  // +2 to include one decimal place
    }
    return theMagFieldStr;
  }

  //_____________________________________________________________________
  inline const std::string apvModeAsString(const SiStripLatency* latency) {
    if (latency) {
      switch (latency->singleReadOutMode()) {
        case k_PeakMode:
          return "PEAK";  // peak mode
        case k_DeconvolutionMode:
          return "DECO";  // deco mode
        default:
          return "UNDEF";  // undefined
      }
    } else {
      return "UNDEF";
    }
  }

  //_____________________________________________________________________
  inline double fitFunction(double* x, double* par) {
    double a = par[0];
    double thetaL = par[1];
    double b = par[2];

    double tanThetaL = std::tan(thetaL);
    double value = a * std::abs(std::tan(x[0]) - tanThetaL) + b;

    //TF1::RejectPoint();  // Reject points outside the fit range
    return value;
  }
}  // namespace siStripLACalibration
#endif
