#ifndef SiStripApvSimulationParameters_h
#define SiStripApvSimulationParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"

namespace CLHEP {
  class HepRandomEngine;
}

/**
 * Stores a histogram binned in PU, z and baseline voltage, for every barrel layer of the strip tracker
 */
class SiStripApvSimulationParameters {
public:
  using layerid = unsigned int;
  using LayerParameters = PhysicsTools::Calibration::HistogramF3D;

  SiStripApvSimulationParameters(layerid nTIB, layerid nTOB, layerid nTID, layerid nTEC)
      : m_nTIB(nTIB), m_nTOB(nTOB), m_nTID(nTID), m_nTEC(nTEC) {
    m_barrelParam.resize(m_nTIB + m_nTOB);
    m_barrelParam_xInt.resize(m_nTIB + m_nTOB);
    m_endcapParam.resize(m_nTID + m_nTEC);
    m_endcapParam_xInt.resize(m_nTID + m_nTEC);
  }
  SiStripApvSimulationParameters() {}
  ~SiStripApvSimulationParameters() {}

  void calculateIntegrals();  // make sure integrals have been calculated

  bool putTIB(layerid layer, const LayerParameters& params) { return putTIB(layer, LayerParameters(params)); }
  bool putTIB(layerid layer, LayerParameters&& params);

  bool putTOB(layerid layer, const LayerParameters& params) { return putTOB(layer, LayerParameters(params)); }
  bool putTOB(layerid layer, LayerParameters&& params);

  bool putTID(layerid wheel, const LayerParameters& params) { return putTID(wheel, LayerParameters(params)); }
  bool putTID(layerid wheel, LayerParameters&& params);

  bool putTEC(layerid wheel, const LayerParameters& params) { return putTEC(wheel, LayerParameters(params)); }
  bool putTEC(layerid wheel, LayerParameters&& params);

  const LayerParameters& getTIB(layerid layer) const { return m_barrelParam[layer - 1]; }
  const LayerParameters& getTOB(layerid layer) const { return m_barrelParam[m_nTIB + layer - 1]; }
  const LayerParameters& getTID(layerid wheel) const { return m_endcapParam[wheel - 1]; }
  const LayerParameters& getTEC(layerid wheel) const { return m_endcapParam[m_nTID + wheel - 1]; }

  float sampleTIB(layerid layer, float z, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleBarrel(layer - 1, z, pu, engine);
  }
  float sampleTOB(layerid layer, float z, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleBarrel(m_nTIB + layer - 1, z, pu, engine);
  };
  float sampleTID(layerid wheel, float r, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleEndcap(wheel - 1, r, pu, engine);
  }
  float sampleTEC(layerid wheel, float r, float pu, CLHEP::HepRandomEngine* engine) const {
    return sampleEndcap(m_nTID + wheel - 1, r, pu, engine);
  }

private:
  layerid m_nTIB, m_nTOB, m_nTID, m_nTEC;
  std::vector<PhysicsTools::Calibration::HistogramF3D> m_barrelParam;
  std::vector<PhysicsTools::Calibration::HistogramF2D> m_barrelParam_xInt;
  std::vector<PhysicsTools::Calibration::HistogramF3D> m_endcapParam;
  std::vector<PhysicsTools::Calibration::HistogramF2D> m_endcapParam_xInt;

  float sampleBarrel(layerid layerIdx, float z, float pu, CLHEP::HepRandomEngine* engine) const;
  float sampleEndcap(layerid wheelIdx, float r, float pu, CLHEP::HepRandomEngine* engine) const;

  COND_SERIALIZABLE;
};

#endif
