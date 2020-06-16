// -*- C++ -*-
// //
// // Package:    RecoPixelVertexing/PixelTriplets
// // Class:      CACut
// //
// // Original Author:  Karla Josefina Pena Rodriguez
// //         Created:  Wed, 14 Feb 2019 10:30:00 GMT
// //

#ifndef RecoPixelVertexing_PixelTriplets_src_CACut_h
#define RecoPixelVertexing_PixelTriplets_src_CACut_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CAGraph.h"

class CACut {
public:
  explicit CACut(const double defaultCut, const std::vector<edm::ParameterSet> &tripletCuts)
      : useCACuts_(true), foundAllLayerIds_(false), defaultCut_(defaultCut) {
    if (tripletCuts.size() == 1 && tripletCuts[0].getParameter<double>("cut") == -1.) {
      useCACuts_ = false;
      LogDebug("Configuration") << "No CACut VPSet. Using default cut value of " << defaultCut << " for all layer triplets";
      return;
    }

    valuesByTripletNames_.reserve(tripletCuts.size());
    valuesByLayerIds_.reserve(tripletCuts.size());

    setCutValuesByTripletNames(tripletCuts);
  }

  void setCutValuesByTripletNames(const std::vector<edm::ParameterSet> &tripletCuts) {
    for (const auto &thisTriplet : tripletCuts) {
      valuesByTripletNames_.emplace_back();
      auto &thisCACut = valuesByTripletNames_.back();

      thisCACut.tripletName = thisTriplet.getParameter<std::string>("seedingLayers");
      thisCACut.cutValue = thisTriplet.getParameter<double>("cut");
    }
  }

  void setCutValuesByLayerIds(CAGraph &caLayers) {
    if (!useCACuts_ || foundAllLayerIds_)
      return;

    foundAllLayerIds_ = true;
    valuesByLayerIds_.clear();

    for (const auto &thisTriplet : valuesByTripletNames_) {
      valuesByLayerIds_.emplace_back();
      auto &thisCACut = valuesByLayerIds_.back();

      // Triplet name, e.g. 'BPix1+BPix2+BPix3'
      std::string layersToSet = thisTriplet.tripletName;
      for (int thisLayer = 0; thisLayer < 3; thisLayer++) {
        // Get layer name
        std::size_t layerPos = layersToSet.find("+");
        if ((thisLayer < 2 && layerPos == std::string::npos) || (thisLayer == 2 && layerPos != std::string::npos)) {
          throw cms::Exception("Configuration")
              << "Please enter a valid triplet name in the CACuts parameter set; e.g. 'BPix1+BPix2+BPix3'";
        }

        std::string layerName = layersToSet.substr(0, layerPos);
        layersToSet = layersToSet.substr(layerPos + 1);

        // Get layer ID
        thisCACut.layerIds.emplace_back(caLayers.getLayerId(layerName));
        if (thisCACut.layerIds.back() == -1) {
          foundAllLayerIds_ = false;
          edm::LogWarning("Configuration")
              << "Layer name '" << layerName << "' not found in the CAGraph. Please check CACuts parameter set.";
        }
      }

      // Cut
      thisCACut.cutValue = thisTriplet.cutValue;
      thisCACut.hasValueByInnerLayerId = false;
    }

    setCutValuesByInnerLayerIds();
  }

  void setCutValuesByInnerLayerIds() {
    for (auto &thisTriplet : valuesByLayerIds_) {
      if (thisTriplet.hasValueByInnerLayerId)
        continue;
      auto it = std::find(thisTriplet.layerIds.begin(), thisTriplet.layerIds.end(), -1);
      if (it != thisTriplet.layerIds.end())
        continue;

      bool foundOuterDoublet = false;

      for (auto &thisOuterDoublet : valuesByInnerLayerIds_) {
        if (thisOuterDoublet.outerDoubletIds[0] == thisTriplet.layerIds[1] &&
            thisOuterDoublet.outerDoubletIds[1] == thisTriplet.layerIds[2]) {
          thisOuterDoublet.innerLayerIds.emplace_back(thisTriplet.layerIds[0]);
          thisOuterDoublet.cutValues.emplace_back(thisTriplet.cutValue);
          foundOuterDoublet = true;
          break;
        }
      }

      if (!foundOuterDoublet) {
        valuesByInnerLayerIds_.emplace_back(defaultCut_);
        auto &newOuterDoublet = valuesByInnerLayerIds_.back();

        newOuterDoublet.outerDoubletIds.emplace_back(thisTriplet.layerIds[1]);
        newOuterDoublet.outerDoubletIds.emplace_back(thisTriplet.layerIds[2]);
        newOuterDoublet.innerLayerIds.emplace_back(thisTriplet.layerIds[0]);
        newOuterDoublet.cutValues.emplace_back(thisTriplet.cutValue);
      }

      thisTriplet.hasValueByInnerLayerId = true;
    }
  }

  struct CAValuesByInnerLayerIds {
    explicit CAValuesByInnerLayerIds(float cut) : defaultCut_(cut) {}

    float at(int layerId) const {
      for (size_t thisLayer = 0; thisLayer < innerLayerIds.size(); thisLayer++) {
        if (innerLayerIds.at(thisLayer) == layerId)
          return cutValues.at(thisLayer);
      }

      return defaultCut_;
    }

    std::vector<int> outerDoubletIds;
    std::vector<int> innerLayerIds;
    std::vector<float> cutValues;

  private:
    double defaultCut_;
  };

  CAValuesByInnerLayerIds getCutsByInnerLayer(int layerIds1, int layerIds2) const {
    for (const auto &thisCut : valuesByInnerLayerIds_) {
      if (thisCut.outerDoubletIds[0] == layerIds1 && thisCut.outerDoubletIds[1] == layerIds2) {
        return thisCut;
      }
    }

    CAValuesByInnerLayerIds emptyCutsByInnerLayer(defaultCut_);
    return emptyCutsByInnerLayer;
  }

private:
  struct CAValueByTripletName {
    std::string tripletName;
    float cutValue;
  };

  struct CAValueByLayerIds {
    std::vector<int> layerIds;
    float cutValue;
    bool hasValueByInnerLayerId;
  };

private:
  std::vector<CAValueByTripletName> valuesByTripletNames_;
  std::vector<CAValueByLayerIds> valuesByLayerIds_;
  std::vector<CAValuesByInnerLayerIds> valuesByInnerLayerIds_;
  bool useCACuts_;
  bool foundAllLayerIds_;
  const float defaultCut_;
};

#endif
