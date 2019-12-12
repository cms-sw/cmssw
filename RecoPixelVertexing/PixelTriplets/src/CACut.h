#ifndef RecoPixelVertexing_PixelTriplets_src_CACut_h
#define RecoPixelVertexing_PixelTriplets_src_CACut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CAGraph.h"

class CACut {
public:
  explicit CACut(const double defaultCut, const std::vector<edm::ParameterSet> &tripletCuts)
      : usingCACuts(true), defaultCut_(defaultCut) {

    if  ( tripletCuts.size() == 1 && tripletCuts[0].getParameter<double>("cut") == -1. ) {
      usingCACuts = false;
      edm::LogWarning("Configuration") << "No CACut VPSet. Using default cut value of " << defaultCut << " for all layer triplets";
      return;
    }

    setCutValuesByTripletNames(tripletCuts);
  }

  void setCutValuesByTripletNames(const std::vector<edm::ParameterSet> &tripletCuts) {
    for (const auto &thisTriplet : tripletCuts) {
      CAValueByTripletName thisCACut;
      thisCACut.tripletName = thisTriplet.getParameter<std::string>("seedingLayers");
      thisCACut.cutValue = thisTriplet.getParameter<double>("cut");

      valuesByTripletNames_.emplace_back(thisCACut);
    }
  }

  void setCutValuesByLayerIds(CAGraph &caLayers) {
    if ( !usingCACuts ) return;
    for (const auto &thisTriplet : valuesByTripletNames_) {
      CAValueByLayerIds thisCACut;

      // Triplet name, e.g. 'BPix1+BPix2+BPix3'
      std::string layersToSet = thisTriplet.tripletName;
      for (int thisLayer = 0; thisLayer < 3; thisLayer++) {

        // Get layer name
        std::size_t layerPos = layersToSet.find("+");
        if ( (thisLayer<2 && layerPos==std::string::npos) || (thisLayer==2 && layerPos!=std::string::npos) ) {
          throw cms::Exception("Configuration")
              << "Please enter a valid triplet name in the CACuts parameter set; e.g. 'BPix1+BPix2+BPix3'";
        }

        std::string layerName = layersToSet.substr(0, layerPos);
        layersToSet = layersToSet.substr(layerPos + 1);

        // Get layer ID
        thisCACut.layerIds.emplace_back(caLayers.getLayerId(layerName));
        if (thisCACut.layerIds.back() == -1) {
          edm::LogWarning("Configuration")
              << "Layer name '" << layerName
              << "' not found in the CAGraph. Please check CACuts parameter set.";
        }
      }

      // Cut
      thisCACut.cutValue = thisTriplet.cutValue;

      // Add to vector
      valuesByLayerIds_.emplace_back(thisCACut);
    }
  }

  class CAValuesByInnerLayerIds {
  public:
    explicit CAValuesByInnerLayerIds(float cut) : defaultCut_(cut) {}

    float at(int layerId) {
      for (size_t thisLayer : layerIds) {
        if (layerIds.at(thisLayer) == layerId)
          return cutValues.at(thisLayer);
      }

      return defaultCut_; // Add LogWarning?
    }

    
    std::vector<int> layerIds;
    std::vector<float> cutValues;

  private:
    double defaultCut_;
  };

  // Check all triplets with outer cell (layerId1, layerId2) and return a map of (layerId0, cut)
  CAValuesByInnerLayerIds getCutsByInnerLayer(int layerIds1, int layerIds2) const {
    CAValuesByInnerLayerIds cutsByInnerLayer(defaultCut_);

    for (const auto &thisCut : valuesByLayerIds_) {
      if (thisCut.layerIds[1] == layerIds1 && thisCut.layerIds[2] == layerIds2) {
        cutsByInnerLayer.layerIds.emplace_back(thisCut.layerIds[0]);
        cutsByInnerLayer.cutValues.emplace_back(thisCut.cutValue);
      }
    }

    return cutsByInnerLayer;
  }

private:
  class CAValueByTripletName {
  public:
    std::string tripletName;
    float cutValue;
  };

  class CAValueByLayerIds {
  public:
    std::vector<int> layerIds;
    float cutValue;
  };

public:
  bool usingCACuts;
private:
  std::vector<CAValueByTripletName> valuesByTripletNames_;
  std::vector<CAValueByLayerIds> valuesByLayerIds_;
  const float defaultCut_;
};

#endif
