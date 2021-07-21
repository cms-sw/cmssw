#ifndef __L1Trigger_L1THGCal_HGCalTriggerTools_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTools_h__

/** \class HGCalTriggerTools
 *  Tools for handling HGCal trigger det-ID: in the current version
 *  of trhe HGCAL simulation only HGCalDetId for the TriggerCells (TC)
 *  are used and not HcalDetId as in the offline!
 *  As a consequence the class assumes that only DetIds of the first kind are used in the getTC* methods
 *  NOTE: this uses the trigger geometry hence would give wrong results
 *  when used for offline reco!!!!
 *
 *  \author G. Cerminara (CERN), heavily "inspired" by HGCalRechHitTools ;)
 */

#include <array>
#include <cmath>
#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

class HGCalTriggerTools {
public:
  HGCalTriggerTools() : geom_(nullptr), eeLayers_(0), fhLayers_(0), bhLayers_(0), noseLayers_(0), totalLayers_(0) {}
  ~HGCalTriggerTools() {}

  void eventSetup(const edm::EventSetup&);
  GlobalPoint getTCPosition(const DetId& id) const;
  unsigned layers(ForwardSubdetector type) const;
  unsigned layers(DetId::Detector type) const;
  unsigned layer(const DetId&) const;
  unsigned layerWithOffset(const DetId&) const;
  bool isEm(const DetId&) const;
  bool isHad(const DetId& id) const { return !isEm(id); }
  bool isSilicon(const DetId&) const;
  bool isScintillator(const DetId& id) const { return !isSilicon(id); }
  bool isNose(const DetId&) const;
  int zside(const DetId&) const;
  // tc argument is needed because of the impossibility
  // to know whether the ID is a TC or a sensor cell
  // in the v8 geometry detid scheme
  int thicknessIndex(const DetId&, bool tc = false) const;

  unsigned lastLayerEE(bool nose = false) const { return (nose ? HFNoseDetId::HFNoseLayerEEmax : eeLayers_); }
  unsigned lastLayerFH() const { return eeLayers_ + fhLayers_; }
  unsigned lastLayerBH() const { return totalLayers_; }
  unsigned lastLayerNose() const { return noseLayers_; }
  unsigned lastLayer(bool nose = false) const { return nose ? noseLayers_ : totalLayers_; }

  // 4-vector helper functions using GlobalPoint
  float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
  float getPhi(const GlobalPoint& position) const;
  float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

  // 4-vector helper functions using DetId
  float getTCEta(const DetId& id, const float& vertex_z = 0.) const;
  float getTCPhi(const DetId& id) const;
  float getTCPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

  inline const HGCalTriggerGeometryBase* getTriggerGeometry() const { return geom_; };

  float getLayerZ(const unsigned& layerWithOffset) const;
  float getLayerZ(const int& subdet, const unsigned& layer) const;

  template <typename T>
  std::vector<T> bxVectorToVector(const BXVector<T>& inputBXVector) {
    std::vector<T> outputVector;
    // loop over collection for a given bx and put the objects into a std::vector
    outputVector.insert(outputVector.end(), inputBXVector.begin(0), inputBXVector.end(0));
    return outputVector;
  }

  DetId simToReco(const DetId&, const HGCalTopology&) const;
  DetId simToReco(const DetId&, const HcalTopology&) const;
  unsigned triggerLayer(const unsigned id) const { return geom_->triggerLayer(id); }

  static constexpr unsigned kScintillatorPseudoThicknessIndex_ = 3;

  enum SubDetectorType {
    hgcal_silicon_CEE,
    hgcal_silicon_CEH,
    hgcal_scintillator,
  };
  SubDetectorType getSubDetectorType(const DetId& id) const;

private:
  const HGCalTriggerGeometryBase* geom_;
  unsigned eeLayers_;
  unsigned fhLayers_;
  unsigned bhLayers_;
  unsigned noseLayers_;
  unsigned totalLayers_;

  int sensorCellThicknessV8(const DetId& id) const;
};

#endif
