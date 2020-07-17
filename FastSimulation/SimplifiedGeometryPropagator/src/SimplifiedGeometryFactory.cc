#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometryFactory.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <TH1F.h>
#include <cctype>
#include <memory>


fastsim::SimplifiedGeometryFactory::SimplifiedGeometryFactory(
    const GeometricSearchTracker *geometricSearchTracker,
    const MagneticField &magneticField,
    const std::map<std::string, fastsim::InteractionModel *> &interactionModelMap,
    double magneticFieldHistMaxR,
    double magneticFieldHistMaxZ)
    : geometricSearchTracker_(geometricSearchTracker),
      magneticField_(&magneticField),
      interactionModelMap_(&interactionModelMap),
      magneticFieldHistMaxR_(magneticFieldHistMaxR),
      magneticFieldHistMaxZ_(magneticFieldHistMaxZ) {
  // naming convention for barrel DetLayer lists
  barrelDetLayersMap_["BPix"] = &geometricSearchTracker_->pixelBarrelLayers();
  barrelDetLayersMap_["TIB"] = &geometricSearchTracker_->tibLayers();
  barrelDetLayersMap_["TOB"] = &geometricSearchTracker_->tobLayers();

  // naming convention for forward DetLayer lists
  forwardDetLayersMap_["negFPix"] = &geometricSearchTracker_->negPixelForwardLayers();
  forwardDetLayersMap_["posFPix"] = &geometricSearchTracker_->posPixelForwardLayers();
  forwardDetLayersMap_["negTID"] = &geometricSearchTracker_->negTidLayers();
  forwardDetLayersMap_["posTID"] = &geometricSearchTracker_->posTidLayers();
  forwardDetLayersMap_["negTEC"] = &geometricSearchTracker_->negTecLayers();
  forwardDetLayersMap_["posTEC"] = &geometricSearchTracker_->posTecLayers();
}

std::unique_ptr<fastsim::BarrelSimplifiedGeometry> fastsim::SimplifiedGeometryFactory::createBarrelSimplifiedGeometry(
    const edm::ParameterSet &cfg) const {
  std::unique_ptr<fastsim::SimplifiedGeometry> layer = createSimplifiedGeometry(BARREL, cfg);
  return std::unique_ptr<fastsim::BarrelSimplifiedGeometry>(
      static_cast<fastsim::BarrelSimplifiedGeometry *>(layer.release()));
}

std::unique_ptr<fastsim::ForwardSimplifiedGeometry> fastsim::SimplifiedGeometryFactory::createForwardSimplifiedGeometry(
    const fastsim::SimplifiedGeometryFactory::LayerType layerType, const edm::ParameterSet &cfg) const {
  if (layerType != NEGFWD && layerType != POSFWD) {
    throw cms::Exception("fastsim::SimplifiedGeometry::createForwardLayer")
        << " called with forbidden layerType. Allowed layerTypes are NEGFWD and POSFWD";
  }
  std::unique_ptr<fastsim::SimplifiedGeometry> layer = createSimplifiedGeometry(layerType, cfg);
  return std::unique_ptr<fastsim::ForwardSimplifiedGeometry>(
      static_cast<fastsim::ForwardSimplifiedGeometry *>(layer.release()));
}

std::unique_ptr<fastsim::SimplifiedGeometry> fastsim::SimplifiedGeometryFactory::createSimplifiedGeometry(
    const fastsim::SimplifiedGeometryFactory::LayerType layerType, const edm::ParameterSet &cfg) const {
  // some flags for internal usage
  bool isForward = true;
  bool isOnPositiveSide = false;
  if (layerType == BARREL) {
    isForward = false;
  } else if (layerType == POSFWD) {
    isOnPositiveSide = true;
  }

  // -------------------------------
  // extract DetLayer (i.e. full geometry of tracker modules)
  // -------------------------------

  std::string detLayerName = cfg.getUntrackedParameter<std::string>("activeLayer", "");
  const DetLayer *detLayer = nullptr;

  if (!detLayerName.empty() && geometricSearchTracker_) {
    if (isForward) {
      detLayerName = (isOnPositiveSide ? "pos" : "neg") + detLayerName;
    }
    detLayer = getDetLayer(detLayerName, *geometricSearchTracker_);
  }

  // ------------------------------
  // radius / z of layers
  // ------------------------------

  // first try to get it from the configuration
  double position = 0;
  std::string positionParameterName = (isForward ? "z" : "radius");
  if (cfg.exists(positionParameterName)) {
    position = fabs(cfg.getUntrackedParameter<double>(positionParameterName));
    if (isForward && !isOnPositiveSide) {
      position = -position;
    }
  }
  // then try extracting from detLayer
  else if (detLayer) {
    if (isForward) {
      position = static_cast<ForwardDetLayer const *>(detLayer)->surface().position().z();
    } else {
      position = static_cast<BarrelDetLayer const *>(detLayer)->specificSurface().radius();
    }
  }
  // then throw error
  else {
    std::string cfgString;
    cfg.allToString(cfgString);
    throw cms::Exception("fastsim::SimplifiedGeometry")
        << "Cannot extract a " << (isForward ? "position" : "radius") << " for this "
        << (isForward ? "forward" : "barrel") << " layer:\n"
        << cfgString;
  }

  // -----------------------------
  // create the layers
  // -----------------------------

  std::unique_ptr<fastsim::SimplifiedGeometry> layer;
  if (isForward) {
    layer = std::make_unique<fastsim::ForwardSimplifiedGeometry>(position);
  } else {
    layer = std::make_unique<fastsim::BarrelSimplifiedGeometry>(position);
  }
  layer->detLayer_ = detLayer;

  // -----------------------------
  // thickness histogram
  // -----------------------------

  // Get limits
  const std::vector<double> &limits = cfg.getUntrackedParameter<std::vector<double> >("limits");
  // and check order.
  for (unsigned index = 1; index < limits.size(); index++) {
    if (limits[index] < limits[index - 1]) {
      std::string cfgString;
      cfg.allToString(cfgString);
      throw cms::Exception("fastsim::SimplifiedGeometryFactory")
          << "limits must be provided in increasing order. error in:\n"
          << cfgString;
    }
  }
  // Get thickness values
  const std::vector<double> &thickness = cfg.getUntrackedParameter<std::vector<double> >("thickness");
  // and check compatibility with limits
  if (limits.size() < 2 || thickness.size() != limits.size() - 1) {
    std::string cfgString;
    cfg.allToString(cfgString);
    throw cms::Exception("fastim::SimplifiedGeometryFactory")
        << "layer thickness and limits not configured properly! error in:" << cfgString;
  }
  // create the histogram
  layer->thicknessHist_ = std::make_unique<TH1F>("h", "h", limits.size() - 1, &limits[0]);
  layer->thicknessHist_->SetDirectory(nullptr);
  for (unsigned i = 1; i < limits.size(); ++i) {
    layer->thicknessHist_->SetBinContent(i, thickness[i - 1]);
  }

  // -----------------------------
  // nuclear interaction thickness factor
  // -----------------------------

  layer->nuclearInteractionThicknessFactor_ =
      cfg.getUntrackedParameter<double>("nuclearInteractionThicknessFactor", 1.);

  // -----------------------------
  // magnetic field
  // -----------------------------

  layer->magneticFieldHist_ = std::make_unique<TH1F>(
      "h", "h", 100, 0., isForward ? magneticFieldHistMaxR_ : magneticFieldHistMaxZ_);
  layer->magneticFieldHist_->SetDirectory(nullptr);
  for (int i = 1; i <= 101; i++) {
    GlobalPoint point = isForward ? GlobalPoint(layer->magneticFieldHist_->GetXaxis()->GetBinCenter(i), 0., position)
                                  : GlobalPoint(position, 0., layer->magneticFieldHist_->GetXaxis()->GetBinCenter(i));
    layer->magneticFieldHist_->SetBinContent(i, magneticField_->inTesla(point).z());
  }

  // -----------------------------
  // list of interaction models
  // -----------------------------

  std::vector<std::string> interactionModelLabels =
      cfg.getUntrackedParameter<std::vector<std::string> >("interactionModels");
  for (const auto &label : interactionModelLabels) {
    std::map<std::string, fastsim::InteractionModel *>::const_iterator interactionModel =
        interactionModelMap_->find(label);
    if (interactionModel == interactionModelMap_->end()) {
      throw cms::Exception("fastsim::SimplifiedGeometryFactory") << "unknown interaction model '" << label << "'";
    }
    layer->interactionModels_.push_back(interactionModel->second);
  }

  // -----------------------------
  // Hack to interface "old" calorimetry with "new" propagation in tracker
  // -----------------------------

  if (cfg.exists("caloType")) {
    std::string caloType = cfg.getUntrackedParameter<std::string>("caloType");

    if (caloType == "PRESHOWER1") {
      layer->setCaloType(SimplifiedGeometry::PRESHOWER1);
    } else if (caloType == "PRESHOWER2") {
      layer->setCaloType(SimplifiedGeometry::PRESHOWER2);
    } else if (caloType == "ECAL") {
      layer->setCaloType(SimplifiedGeometry::ECAL);
    } else if (caloType == "HCAL") {
      layer->setCaloType(SimplifiedGeometry::HCAL);
    } else if (caloType == "VFCAL") {
      layer->setCaloType(SimplifiedGeometry::VFCAL);
    } else {
      throw cms::Exception("fastsim::SimplifiedGeometryFactory")
          << "unknown caloType '" << caloType << "' (defined PRESHOWER1, PRESHOWER2, ECAL, HCAL, VFCAL)";
    }
  }

  // -----------------------------
  // and return the layer!
  // -----------------------------

  return layer;
}

const DetLayer *fastsim::SimplifiedGeometryFactory::getDetLayer(
    const std::string &detLayerName, const GeometricSearchTracker &geometricSearchTracker) const {
  if (detLayerName.empty()) {
    return nullptr;
  }

  // obtain the index from the detLayerName
  unsigned pos = detLayerName.size();
  while (isdigit(detLayerName[pos - 1])) {
    pos -= 1;
  }
  if (pos == detLayerName.size()) {
    throw cms::Exception("fastsim::SimplifiedGeometry::getDetLayer")
        << "last part of detLayerName must be index of DetLayer in list. Error in detLayerName" << detLayerName
        << std::endl;
  }
  int index = atoi(detLayerName.substr(pos).c_str());
  std::string detLayerListName = detLayerName.substr(0, pos);

  try {
    // try to find the detLayer in the barrel map
    if (barrelDetLayersMap_.find(detLayerListName) != barrelDetLayersMap_.end()) {
      auto detLayerList = barrelDetLayersMap_.find(detLayerListName)->second;
      return detLayerList->at(index - 1);  // use at, to provoce the throwing of an error in case of a bad index
    }

    // try to find the detLayer in the forward map
    else if (forwardDetLayersMap_.find(detLayerListName) != forwardDetLayersMap_.end()) {
      auto detLayerList = forwardDetLayersMap_.find(detLayerListName)->second;
      return detLayerList->at(index - 1);  // use at, to provoce the throwing of an error in case of a bad index
    }
    // throw an error
    else {
      throw cms::Exception("fastsim::SimplifiedGeometry::getDetLayer")
          << " could not find list of detLayers corresponding to detLayerName " << detLayerName << std::endl;
    }
  } catch (const std::out_of_range &error) {
    throw cms::Exception("fastsim::SimplifiedGeometry::getDetLayer")
        << " index out of range for detLayerName: " << detLayerName << " " << error.what() << std::endl;
  }
}
